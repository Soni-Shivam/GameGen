import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider="fireworks-ai",
    api_key=hf_token,
)

user_input = input()
system_prompt = """
You are a game designer assistant. Your job is to convert a user’s description of desired gameplay into structured JSON.

Extract and return the following JSON:
{
  "speed_multiplier": float,
  "gravity": float,
  "object_spawn_rate": float,
  "gap_size": int,
  "difficulty_level": string (Easy, Medium, Hard),
  "additional_flags": [string]
}

Only return valid JSON without any extra explanation.
"""

# Send prompt to the model
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ],
)

# Print the JSON response
print(completion.choices[0].message['content'])
