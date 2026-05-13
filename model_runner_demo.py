"""
Part 1 – Docker Model Runner demo
Model image : ai/smollm2:360M-instruct-q4_K_M
Endpoint    : http://localhost:12434/engines/llama.cpp/v1/chat/completions
              (OpenAI-compatible chat completions API)

Before running this script:
    docker model pull ai/smollm2:360M-instruct-q4_K_M
    docker model run  ai/smollm2:360M-instruct-q4_K_M
"""

import json
import requests

ENDPOINT = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"
MODEL_ID  = "ai/smollm2:360M-instruct-q4_K_M"

QUESTIONS = [
    "Explain what Docker is in one sentence.",
    "What is a container image?",
]


def query(prompt: str) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def main():
    print(f"Docker Model Runner – {MODEL_ID}")
    print(f"Endpoint: {ENDPOINT}\n")

    for prompt in QUESTIONS:
        print(f"Q: {prompt}")
        answer = query(prompt)
        print(f"A: {answer}\n")
        print("-" * 60 + "\n")

    print("Raw JSON for last response (for README documentation):")
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": QUESTIONS[-1]}],
    }
    raw = requests.post(ENDPOINT, json=payload, timeout=60)
    print(json.dumps(raw.json(), indent=2))


if __name__ == "__main__":
    main()
