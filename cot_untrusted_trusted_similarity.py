import os
import requests

from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

def generate_from_cot(
    cot_text: str,
    model: str,
    hf_token: str,
    max_new_tokens: int = 250,
    temperature: float = 0.05
):
    client = InferenceClient(api_key=HF_TOKEN)

    prompt = (
        "You are a helpful and accurate AI assistant."
        "Below is a chain of thought describing reasoning about a problem.\n"
        "Based ONLY on this chain of thought, produce the final answer as an LLM would.\n\n"
        "Chain of Thought:\n"
        f"{cot_text}\n\n"
        "The final ANSWER based on the given chain of thought:"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    return completion.choices[0].message["content"].strip()