"""
Run this after setting your API key:
  Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
  Then: python test_llm.py
"""
import anthropic
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key.startswith("sk-ant-your"):
    raise SystemExit("Set ANTHROPIC_API_KEY environment variable before running.")

client = anthropic.Anthropic(api_key=api_key)

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is Docker in one sentence?"}
    ],
)
print(message.content[0].text)
