from openai import OpenAI

import config

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_response(prompt, temperature=config.a_temperature):
    print(prompt)
    chat_response = client.chat.completions.create(
        model=config.model_name_or_path,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stop=["<|eot_id|>", "\n\n---", "\n\n\n", "---"],
        temperature=temperature,
        max_tokens=4096,
    )
    response = chat_response.choices[0].message.content
    return response


def get_completion(prompt, temperature=config.q_temperature):
    chat_response = client.completions.create(
        model=config.model_name_or_path,
        prompt=prompt,
        temperature=temperature,
        max_tokens=1024,
        stop=["<|eot_id|>", "\n\n---", "\n\n\n", "---"]
    )
    response = chat_response.choices[0]
    return response
