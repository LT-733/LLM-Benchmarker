import sklearn
import get_outputs, output_judge, clustering
import os
import sys
import dotenv
import torch
import transformers
import torch.nn as nn
dotenv.load_dotenv("./.env")

def main():
    q_and_a: list = get_outputs.get_question_and_answer()
    indexed_models: dict = get_outputs.get_available_models()
    chat_outputs: list[dict] = get_outputs.get_chat_content(q_and_a[0])
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    model = transformers.AutoModel.from_pretrained("sentence-transformers/all-miniLM-L6-V2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-miniLM-L6-V2")
    semantics = output_judge.get_semantic_drift(tst_model=model, given_tokenizer=tokenizer, outputs=chat_outputs, ans=q_and_a[1])
    model_data = semantics["drifts"][1]
    gram_schmidt_data: list = [output_judge.gramschmidt_process(embedding_matrix=cur[2].cpu().numpy().tolist(), device=device) for cur in semantics]
    similarity_in_models, similarity_to_answer = output_judge.get_distance_matrix(chat_outputs, q_and_a[1])
    clustered_data = clustering.cluster_result(similarity_in_models)