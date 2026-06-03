"""
Part 1 — Function Calling Basics (standalone, no server or database required)
=============================================================================

This script demonstrates the raw mechanics of function calling with the
Anthropic Messages API. The single most important idea:

    My code executes the function, not the model.
    The model only DECIDES *which* tool to call and with *what* arguments.
    I run it and feed the result back so the model can answer.

The full flow shown below:

    user message
        -> model decides which tool to call (stop_reason == "tool_use")
            -> My code runs the function with the model's arguments
                -> I append a tool_result and call the API again
                    -> model produces the final natural-language answer

Run it:
    # PowerShell
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python function_calling_demo.py
"""

import ast
import json
import operator
import os

import anthropic

MODEL = "claude-haiku-4-5-20251001"

# This file was ENTIRELY generated using claude, but I do know what's going on.

# ---------------------------------------------------------------------------
# 1. The actual Python functions. These are plain functions — nothing about
#    them is "AI". They are what my code runs when the model asks.
# ---------------------------------------------------------------------------

# A tiny in-memory "database" so search_items has something to find.
_FAKE_ITEMS = [
    {"name": "Python Crash Course", "description": "Beginner book on the Python language"},
    {"name": "Steel Longsword", "description": "A finely balanced one-handed blade"},
    {"name": "Tower Shield", "description": "Heavy iron shield for blocking blows"},
    {"name": "PyTorch Cookbook", "description": "Recipes for deep learning in Python"},
]

# Safe arithmetic evaluator (never use bare eval() on model output).
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def get_weather(city: str) -> dict:
    """Pretend weather service — returns canned data so the demo is deterministic."""
    canned = {
        "salt lake city": {"temp_f": 72, "conditions": "sunny"},
        "london": {"temp_f": 55, "conditions": "rainy"},
        "tokyo": {"temp_f": 68, "conditions": "partly cloudy"},
    }
    return canned.get(city.lower(), {"temp_f": 65, "conditions": "clear"})


def calculate(expression: str) -> dict:
    """Evaluate a simple arithmetic expression, e.g. '12 * (3 + 4)'."""
    value = _safe_eval(ast.parse(expression, mode="eval").body)
    return {"expression": expression, "result": value}


def search_items(query: str) -> dict:
    """Search the fake item list by case-insensitive keyword."""
    q = query.lower()
    hits = [it for it in _FAKE_ITEMS if q in it["name"].lower() or q in it["description"].lower()]
    return {"query": query, "count": len(hits), "results": hits}


# Map tool name -> Python callable. The dispatcher uses this.
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_items": search_items,
}


# ---------------------------------------------------------------------------
# 2. The tool SCHEMAS. This is what we send to the model so it knows what
#    tools exist, what they do, and what arguments they take.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression and return the numeric result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '12 * (3 + 4)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_items",
        "description": "Search for items in the database by keyword.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keyword to search for"}},
            "required": ["query"],
        },
    },
]


def run_conversation(client: anthropic.Anthropic, user_message: str) -> str:
    """Run one full request -> tool -> result -> final-answer cycle and narrate it."""
    print("=" * 72)
    print(f"[USER] {user_message}")
    print("=" * 72)

    messages = [{"role": "user", "content": user_message}]

    # ---- Phase 1: send the message + tool schemas; let the model decide ----
    response = client.messages.create(
        model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
    )
    print(f"[MODEL] stop_reason = {response.stop_reason}")

    # If the model answered directly without a tool, we're done.
    if response.stop_reason != "tool_use":
        answer = "".join(b.text for b in response.content if b.type == "text")
        print(f"[MODEL -> FINAL] {answer}\n")
        return answer

    # ---- Phase 2: inspect tool_use blocks and run them in OUR code ----
    # The assistant turn must be appended verbatim before we reply with results.
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"[MODEL DECIDES] call {block.name}({json.dumps(block.input)})")
            try:
                # >>> THIS is the key line: our code runs the function. <<<
                output = TOOL_FUNCTIONS[block.name](**block.input)
                print(f"[YOUR CODE RUNS] {block.name} -> {json.dumps(output)}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output),
                })
            except Exception as exc:  # return the error as words, not a crash
                print(f"[YOUR CODE ERROR] {exc}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {exc}",
                    "is_error": True,
                })

    # ---- Phase 3: send the tool result(s) back for a final answer ----
    messages.append({"role": "user", "content": tool_results})
    final = client.messages.create(
        model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
    )
    answer = "".join(b.text for b in final.content if b.type == "text")
    print(f"[MODEL -> FINAL] {answer}\n")
    return answer


def _load_api_key() -> str | None:
    """Find the Anthropic key.

    Prefer the project's .env (where this app stores the key) over the OS
    environment variable, so the demo uses the same key the rest of the app does
    even if a stale ANTHROPIC_API_KEY is left set in the shell.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for env_path in (os.path.join(here, ".env"), os.path.join(here, os.pardir, ".env")):
        if not os.path.isfile(env_path):
            continue
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    # ignore the .env.example placeholder values
                    if val and not val.lower().startswith(("key", "sk-ant-your")):
                        return val
    return os.getenv("ANTHROPIC_API_KEY")


def main():
    api_key = _load_api_key()
    if not api_key:
        raise SystemExit(
            "No API key found. Put a real ANTHROPIC_API_KEY in fast_api_project/.env "
            "(or set it as an environment variable) before running this demo."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Three prompts, each steering the model toward a different tool.
    run_conversation(client, "What's the weather in Tokyo right now?")
    run_conversation(client, "What is 12 * (3 + 4)?")
    run_conversation(client, "Do we have any items about Python in the database?")


if __name__ == "__main__":
    main()
