# This file was mostly generated used Claude Opus 4.8 (I really wanted to try out the new Opus 4.8 model to see how good it was.

"""
Part 2 — The Agent (Path A: built from scratch on the raw Anthropic SDK)
========================================================================

This module wraps the function-calling mechanics from `function_calling_demo.py`
in a *loop*. The model is given three tools that talk to THIS app's own HTTP
API, and it keeps calling tools until it has enough information to answer.

    task
      -> model decides tool      (search_items / create_item / query_knowledge_base)
        -> we run it (HTTP call to our own API)
          -> feed the result back
            -> model decides the next tool, or gives a final answer
                  (looped, up to AGENT_MAX_STEPS)

Three guardrails are baked in:
  1. Max iterations      - the loop is capped (AGENT_MAX_STEPS).
  2. Tool confirmation   - create_item / delete_item PAUSE the loop and wait
                           for explicit user approval before they run.
  3. Error handling      - tool failures become plain-text tool_results
                           (is_error=True) so the model can recover instead
                           of the loop crashing.

Because confirmation pauses execution across HTTP requests, an agent run is
modelled as a *session* held in memory:
    POST /agent          -> start a session, run until done or a pause
    POST /agent/confirm  -> approve/deny the pending action, resume the session
"""

import json
import os
import re
import uuid

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The agent calls the app's OWN endpoints over HTTP. Inside Docker the backend
# listens on 0.0.0.0:9000, so localhost:9000 resolves to itself.
API_BASE = os.environ.get("AGENT_API_BASE", "http://localhost:9000")
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "10"))

# Guardrail #2: these tools mutate data and must be approved before running.
DESTRUCTIVE_TOOLS = {"create_item", "delete_item"}

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

SYSTEM_PROMPT = (
    "You are an autonomous assistant for the Item Manager app. You help the user "
    "manage a collection of inventory items by using the tools available to you.\n\n"
    "Tools:\n"
    "  - search_items: look up existing items by keyword.\n"
    "  - create_item: add a new item to the collection (requires user approval).\n"
    "  - query_knowledge_base: ask a natural-language question about the collection "
    "(retrieval-augmented; good for summaries and 'what do we have about X').\n\n"
    "Rules:\n"
    "  - For compound tasks such as 'find items about X, and if none exist create one', "
    "FIRST call search_items. Only create an item if the search returns no relevant match.\n"
    "  - Never create duplicate items. Check first.\n"
    "  - When the task is complete, reply with a short plain-text summary of what you "
    "did and found. Do not call any more tools once you are done.\n"
    "  - If a tool returns an error, read it and either adjust your arguments or explain "
    "the problem to the user. Do not repeat the same failing call."
)


# ---------------------------------------------------------------------------
# Tool schemas (what we advertise to the model)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "search_items",
        "description": "Search the item collection by keyword. Returns matching items "
                       "(id, name, description). Use this before creating to avoid duplicates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword(s) to search names and descriptions for."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_item",
        "description": "Create a new item in the collection. This MODIFIES the database and "
                       "requires user confirmation before it runs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short item name."},
                "description": {"type": "string", "description": "Optional longer description."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "query_knowledge_base",
        "description": "Ask a natural-language question about the contents of the item "
                       "collection. Uses retrieval-augmented generation over the items "
                       "(the app's /ask endpoint). Best for summaries and open questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The natural-language question."}
            },
            "required": ["question"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations — each calls THIS app's own HTTP API
# ---------------------------------------------------------------------------
def _tool_search_items(query: str) -> dict:
    resp = requests.get(f"{API_BASE}/items", params={"q": query}, timeout=30)
    resp.raise_for_status()
    items = resp.json()
    return {"query": query, "count": len(items), "items": items}


def _tool_create_item(name: str, description: str | None = None) -> dict:
    resp = requests.post(
        f"{API_BASE}/items",
        json={"name": name, "description": description},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _tool_query_knowledge_base(question: str) -> dict:
    resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


TOOL_IMPL = {
    "search_items": _tool_search_items,
    "create_item": _tool_create_item,
    "query_knowledge_base": _tool_query_knowledge_base,
}


def _execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool. Returns (content_string, is_error). Guardrail #3: never raise."""
    fn = TOOL_IMPL.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'.", True
    try:
        output = fn(**tool_input)
        return json.dumps(output, default=str), False
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        body = exc.response.text if exc.response is not None else str(exc)
        return f"Error calling {name}: HTTP {status} - {body}", True
    except requests.RequestException as exc:
        return f"Error calling {name}: the app's API is unreachable ({exc}).", True
    except TypeError as exc:
        return f"Error calling {name}: bad arguments ({exc}).", True
    except Exception as exc:  # noqa: BLE001 - last-resort guardrail
        return f"Error calling {name}: {exc}", True


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
class _AgentSession:
    def __init__(self, task: str, max_steps: int):
        self.id = uuid.uuid4().hex
        self.task = task
        self.max_steps = max_steps
        self.messages: list = [{"role": "user", "content": task}]
        self.steps: list[dict] = []          # ordered reasoning trace
        self.status = "running"              # running|awaiting_confirmation|done|max_steps|error
        self.result: str | None = None
        self.pending: dict | None = None     # {tool_use_id, tool, input} awaiting approval
        self.step_count = 0
        # Mid-turn bookkeeping so we can resume after a confirmation pause.
        self._assistant_content = None       # the assistant content blocks being processed
        self._results: dict = {}             # tool_use_id -> tool_result block


# In-memory session store. Fine for a single-process demo; see README tradeoffs.
_SESSIONS: dict[str, _AgentSession] = {}


def _record(session: _AgentSession, kind: str, **data):
    session.steps.append({"type": kind, **data})


def _snapshot(session: _AgentSession) -> dict:
    """The JSON returned to the client: result + full ordered reasoning trace."""
    return {
        "session_id": session.id,
        "status": session.status,
        "result": session.result,
        "pending_action": session.pending,  # set when status == awaiting_confirmation
        "steps": session.steps,
    }


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
def _continue_turn(session: _AgentSession) -> bool:
    """Process the tool_use blocks of the current assistant turn.

    Returns True when every block has a result and the tool_result user message
    has been appended; returns False if we paused to await confirmation.
    """
    tool_blocks = [b for b in session._assistant_content if b.type == "tool_use"]
    for block in tool_blocks:
        if block.id in session._results:
            continue  # already handled (e.g. resumed after confirmation)

        if block.name in DESTRUCTIVE_TOOLS:
            # Guardrail #2: pause and surface the intended call for approval.
            session.pending = {"tool_use_id": block.id, "tool": block.name, "input": block.input}
            session.status = "awaiting_confirmation"
            _record(session, "confirmation_required", tool=block.name, input=block.input)
            return False

        content, is_error = _execute_tool(block.name, block.input)
        _record(session, "tool", tool=block.name, input=block.input, output=content, error=is_error)
        session._results[block.id] = {
            "type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error,
        }

    # All blocks resolved -> reply to the model with one tool_result user turn.
    results = [session._results[b.id] for b in tool_blocks]
    session.messages.append({"role": "user", "content": results})
    session._assistant_content = None
    session._results = {}
    return True


def _drive(session: _AgentSession) -> None:
    """Run the loop until the agent finishes, hits the step cap, or pauses."""
    while session.status == "running":
        # Resuming mid-turn after a confirmation? Finish that turn first.
        if session._assistant_content is not None:
            if not _continue_turn(session):
                return  # paused again (e.g. a second destructive call)
            continue

        # Guardrail #1: hard cap on iterations.
        if session.step_count >= session.max_steps:
            session.status = "max_steps"
            session.result = "Step limit reached before the task was completed."
            _record(session, "limit", message=session.result)
            return

        session.step_count += 1
        try:
            response = _client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=session.messages,
            )
        except anthropic.APIError as exc:
            session.status = "error"
            session.result = f"LLM error: {exc}"
            _record(session, "error", message=session.result)
            return

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if text:
            _record(session, "thought", text=text)

        if response.stop_reason != "tool_use":
            session.status = "done"
            session.result = text or "(the agent returned no text)"
            return

        # Tool calls requested: append the assistant turn and process its blocks.
        session.messages.append({"role": "assistant", "content": response.content})
        session._assistant_content = response.content
        session._results = {}
        if not _continue_turn(session):
            return  # paused for confirmation


# ---------------------------------------------------------------------------
# Public API (called by the FastAPI endpoints)
# ---------------------------------------------------------------------------
def run_agent_task(task: str, max_steps: int | None = None) -> dict:
    """Start a new agent session and run it until done / paused / capped."""
    session = _AgentSession(task, max_steps or MAX_STEPS)
    _SESSIONS[session.id] = session
    _drive(session)
    return _snapshot(session)


def confirm_action(session_id: str, approved: bool) -> dict:
    """Approve or deny a pending destructive action, then resume the session."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError("Unknown or expired agent session.")
    if session.status != "awaiting_confirmation" or session.pending is None:
        raise ValueError("This session is not waiting for a confirmation.")

    pending = session.pending
    if approved:
        content, is_error = _execute_tool(pending["tool"], pending["input"])
        _record(session, "tool", tool=pending["tool"], input=pending["input"],
                output=content, error=is_error, confirmed=True)
    else:
        # Guardrail #2 (deny path): tell the model in words so it can adapt.
        content = (f"The user DECLINED to approve calling {pending['tool']}. "
                   f"Do not retry it; continue without it or finish.")
        is_error = False
        _record(session, "tool", tool=pending["tool"], input=pending["input"],
                output="(declined by user)", error=False, declined=True)

    session._results[pending["tool_use_id"]] = {
        "type": "tool_result", "tool_use_id": pending["tool_use_id"],
        "content": content, "is_error": is_error,
    }
    session.pending = None
    session.status = "running"
    _drive(session)
    return _snapshot(session)
