models: list = ['tencent/hy3:free']
import get_outputs
import dotenv
import os
dotenv.load_dotenv("./.env")
output = get_outputs.get_chat_content(question="How do you spell cat?", chosen_models=models, API_key=str(os.getenv("API_key")))
print(output)