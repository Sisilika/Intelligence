import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("KEY LOADED:", OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else "NONE")
def ask_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You answer using only provided document context."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    print("\n========== OPENROUTER RAW RESPONSE ==========")
    print(response.text)
    print("============================================\n")

    resp_json = response.json()

    if "choices" not in resp_json:
        return str(resp_json)

    return resp_json["choices"][0]["message"]["content"]
print("KEY LOADED:", OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else "NONE")
