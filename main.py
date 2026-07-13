import webview
import get_outputs, clustering, output_judge
import numpy as np
import keyring
import sys, os

def get_build_path(relative_path: str):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class Bridge():
    def __init__(self):
        import torch, transformers
        import sentence_transformers
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
        self.model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device=self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device, similarity_fn_name="cosine")

    def save_api_key(self, api_key: str) -> bool:
        if not api_key.startswith("sk-or-v1"):
            return False
        keyring.set_password("LLM-Benchmarker", "API_key", api_key)
        return True

    def get_api_key(self) -> dict:
        key = (keyring.get_password("LLM-Benchmarker", "API_key"))
        if key is None:
            return {"success": False, "error message": "There is no API key. Have you forgotten to set it in your settings?"}
        return {
            "success": True,
            "API_key": key
        }


    def get_models(self)->list:
        available_models: list = get_outputs.get_available_models_list()
        return available_models

    def return_outputs(self, question: str, baseline_ans: str, model_list: list[str], numClusters: int):
        try:
            API_key = self.get_api_key()
            if "API_key" not in API_key:
                return API_key
            else:
                API_key = API_key["API_key"]
            model_outputs, errors = get_outputs.get_chat_content(question=question, chosen_models=model_list, API_key=API_key)
            drifts = output_judge.get_semantic_drift(device=self.device, tst_model=self.model, given_tokenizer=self.tokenizer, outputs=model_outputs, ans=baseline_ans)
            novelty_lists: list[dict] = []
            debug_fig = drifts["figure"]
            # returning two things at once sure is annoying I am using only one from now on
            drifts = drifts["drifts"]
            for drift in drifts:
                novelty_lists.append({drift[0]: output_judge.gramschmidt_process(drift[2].cpu().numpy().tolist(), device=self.device)})
            token_similarity: list[list] = [[drift[0], drift[1]] for drift in drifts]
            similarity_between_models, accuracy = output_judge.get_distance_matrix(encoder=self.encoder, model_out=model_outputs, std_ans=baseline_ans)
            scattered_data, labels = clustering.cluster_result(similarity_between_models, k=numClusters)
            scattered_data = scattered_data.tolist()
            labels = labels.tolist()
            if errors == []:
                return {
                    "success": True,
                    "text_outputs": model_outputs,
                    "semantic novelties": novelty_lists,
                    "token_similarities": token_similarity, 
                    "clustering_results": {
                            "MDS_results": scattered_data, 
                            "Agglomerative_coloring": labels
                        }, 
                    "accuracy_scores": accuracy
                }
            else:
                return {
                    "success": True,
                    "text_outputs": model_outputs,
                    "semantic novelties": novelty_lists,
                    "token_similarities": token_similarity, 
                    "clustering_results": {
                            "MDS_results": scattered_data, 
                            "Agglomerative_coloring": labels
                        }, 
                    "accuracy_scores": accuracy,
                    "OpenRouter Errors": errors
                }
        except Exception as e:
            return {
                "success": False,
                "error message":str(e)
            }

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    anything = Bridge()

    if getattr(sys, 'frozen', False):
        entry_point = get_build_path('frontend/dist/index.html')
    else:
        entry_point = 'http://localhost:5173'

    webview.create_window(title="Benchmark", js_api=anything, url=entry_point, width=1200, height=800)
    webview.start(debug=not getattr(sys, 'frozen', False), icon="frontend/src/assets/icon.svg")