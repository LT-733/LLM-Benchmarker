import openrouter
import requests
import json
import os
import time
from openrouter import OpenRouter
import dotenv
dotenv.load_dotenv("./.env")

def get_question_and_answer() -> list[str]:
    user_msg: str = str(input())
    baseline_answer: str = str(input())
    return [user_msg, baseline_answer]

def get_available_models() -> dict:
    free_models = []
    response = requests.get(url="https://openrouter.ai/api/v1/models?extra=free&output_modalities=text")
    response = response.json()

    free_models=[r["id"] for r in response["data"] if (float(r["pricing"]["prompt"]) == 0 and r["architecture"]["modality"]=="text->text")]
    index: int = 1
    model_lookup: dict = {}
    for model in free_models:
        model_lookup[index] = model
        index += 1
    return model_lookup

def get_chat_content(question: str="") -> list[dict]:
    user_msg: str = str(input("Ask a question to our models!\n"))
    # baseline_ans: str = str(input("Provide a baseline answer!\n"))
    model_responses: list[dict] = []

    with OpenRouter(api_key=os.getenv("API_key")) as client:
        model_lookup = get_available_models()
        
        print("you have the access to the following models:")
        for x, y in model_lookup.items():
            print(f"{x}: {y}")
        print("select up to five of them by their indicies: ")
        n = int(input("first off, how many would you like to select?\n"))
        lookup_idx: list = []
        print("Now, select the models you want to use by indexing them.\n")
        for i in range(n):
            lookup_idx.append(int(input()))
        chosen_models: list = []
        for idx in lookup_idx:
            try:
                chosen_models.append(model_lookup[idx])
            except Exception as e:
                print(f"Error: {e}")
        print(chosen_models)
        time.sleep(5)
        for name in chosen_models:
            try:
                curres = client.chat.send(
                    model=name,
                    messages= [
                        {"role": "system", "content": "Answer the questions concisely and precisely. You are to use only plain text, do NOT use markdown or LaTex."},
                        {"role": "user", "content": user_msg if question=="" else question}
                    ],
                )
                model_responses.append({"model": name, "text": curres.choices[0].message.content})
            except Exception as e:
                print(f"error occured: {e}")
                model_responses.append({"model": name, "test": str(e)})
        print(model_responses)
    return model_responses
    
