import numpy as np
import clustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import get_outputs
import math


# test content, will not be used in production
baseline_ans: str = "In computer science, a linked list is a linear collection of data elements whose order is not given by their physical placement in memory. Instead, each element points to the next. It is a data structure consisting of a collection of nodes which together represent a sequence. In its most basic form, each node contains data, and a reference (in other words, a link) to the next node in the sequence. This structure allows for efficient insertion or removal of elements from any position in the sequence during iteration."

# This runs the gram-schmidt process on the output we pulled from the second-to-last layer of the embedding model
# and outputs the spike of new semantic tokens as the model provides output, with a spike being more new concepts/topics being generated
# and low points being reused tokens and semantic stagnation
def gramschmidt_process(embedding_matrix: list[list], device, noise_threshold: float=1e-5):
    """Takes the regular embedding matrix pulled from miniLM/L6/V2 (must be squeezed to be a 2D matrix)

      and your hardware acceleration device of choice
      
      returns the lists of new semantic spikes on token outputs"""
    import torch
    orthogonal_basis: list = []
    results: list[float] = []
    for t in range(len(embedding_matrix)):
        if t == 0:
            curvector = torch.tensor(embedding_matrix[t]).to(device=device)
            results.append(torch.norm(curvector, dim=0).item())
            orthogonal_basis.append(torch.nn.functional.normalize(curvector, dim=0))
        else:
            curvector = torch.tensor(embedding_matrix[t]).to(device=device)
            bases_matrix = torch.stack(orthogonal_basis)
            projection_results = bases_matrix @ curvector
            parallel_vector = bases_matrix.T @ projection_results
            residual = curvector - parallel_vector
            norm_res = torch.norm(residual, dim=0).item()
            results.append(norm_res)
            if norm_res > noise_threshold:
                orthogonal_basis.append(torch.nn.functional.normalize(residual, dim=0))
            else:
                orthogonal_basis.append(torch.zeros(384, dtype=curvector.dtype, device=device))
    return results

# This uses our autotokenizer taken from huggingface and throw the tokenized text outputs into the embedding model
# and it does some cosine similarity score stuff with the embedded baseline output
def get_semantic_drift(device, tst_model, given_tokenizer, outputs: list[dict], ans: str) -> dict:
    """Takes the model (in this case miniLM/L6/V2), and a given tokenizer (in this case Huggingface's Autotokenizer), and the text output you got from your favorite LLMs and your baseline answer that you want those outputs to match

       Does some cosine similarity calculation token by token

       Returns a dictionary consisting of the following:
       {
           a matplotlib figure object that is a heatmap which matches the semantic similarity token by token between each LLM's outputs and the answer

           a list of tuple that has the name of each LLM and the final semantic matching data used to plot the heat map, alongside the raw N by 384 matrix used for the gram schmidt process (in that order!)
       }"""
    import torch
    activation_storage: dict = {}
    target_layer = tst_model.encoder.layer[5]
    model_names = [output['model'] for output in outputs]
    def callback_function(module, in_tensor, out_tensor, model_name: str):
        activation_storage[model_name] = out_tensor.detach().cpu()
    # This is a placeholder we will fill in the PyTorch loop
    curmodel: str = ""

    with torch.no_grad():
        # first we extract the last layer of the embedding model to get the tensor coming out of it, so we can compare the semantic drift from the answer.
        for candidate in outputs:
            curmodel = candidate["model"]
            hook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
            tokenized = given_tokenizer(candidate["text"], return_tensors="pt", truncation=True, max_length=512)
            tokenized = tokenized.to(device=device) # send the memory to the accelerator
            # that double asterisk distributes the key values pairs for you
            _ = tst_model(**tokenized)
            hook.remove()
            del _
        # must also get the embedding of the final layer for our baseline answer
        curmodel = "answer"
        anshook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
        ans_tokenized = given_tokenizer(ans, return_tensors="pt", truncation=True, max_length=512)
        ans_tokenized = ans_tokenized.to(device=device)
        _ = tst_model(**ans_tokenized)
        anshook.remove()
        del _
    
    activation_storage["answer"] = activation_storage["answer"].squeeze(0)
    activation_storage["answer"] = torch.nn.functional.normalize(activation_storage["answer"])
    final_drifts: list[tuple[str, list, list]] = []
    for name, data in (activation_storage).items():
        if name == "answer": 
            continue
        else:
            data = data.squeeze(0)
            tokenmap = data
            data = torch.nn.functional.normalize(data)
            final = (data @ activation_storage["answer"].T)
            final = final.tolist()
            final_drifts.append((name, final, tokenmap))
    fig, axs = plt.subplots(ncols=len(final_drifts), layout="constrained", nrows=1, squeeze=False, figsize=(len(final_drifts)*3.0, 2.5))
    fig.suptitle(f"Semantic Drift From the Baseline Answer of {len(final_drifts)} Models")
    for i in range(len(final_drifts)):
        axs[0, i].imshow(final_drifts[i][1], cmap="viridis", vmin=0.7, vmax=1.0, aspect="auto")
        axs[0, i].set_title(final_drifts[i][0])
        axs[0, i].set_xlabel("Baseline")
        if i == 0:
            axs[0, i].set_ylabel("Response")
    return {
        "figure": fig,
        "drifts": final_drifts
    }

def get_distance_matrix(encoder, model_out: list[dict], std_ans: str) -> tuple[list[list], list]:
    """The easiest of output_judge
      does a general cosine similarity score between the overall embedding output of each LLM 
      and the overall embedding output of the baseline answer"""

    from sentence_transformers import SentenceTransformer, SimilarityFunction
    embeddings: list[torch.Tensor] = []

    for output in model_out:
        embedding = encoder.encode(output["text"], convert_to_tensor=True)
        embeddings.append(embedding)
    embedded_ans: torch.Tensor = encoder.encode(std_ans, convert_to_tensor=True)
    distances: list[list] = []
    distances_to_ans = []
    for i in range(len(embeddings)):
        toans = encoder.similarity(embeddings[i], embedded_ans).item()
        distances_to_ans.append(toans)
        cur = []
        for j in range(len(embeddings)):
            if j != i:
                tocur = 1.0 - encoder.similarity(embeddings[i], embeddings[j]).item()
                cur.append(tocur)
            else:
                cur.append(0.0)
        distances.append(cur)
    return distances, distances_to_ans

# Testing function, will not be used in production
if __name__ == "__main__":
    import torch, transformers
    from sentence_transformers import SentenceTransformer, SimilarityFunction
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    print(f"using {device} as performance accelerator")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device, similarity_fn_name=SimilarityFunction.COSINE)
    model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    model = model.to(device=device)

    text_outputs: list[dict] = get_outputs.get_chat_content()[0]
    dists, _ = get_distance_matrix(encoder=encoder, model_out=text_outputs, std_ans=baseline_ans)
    f, (fig1, fig2) = plt.subplots(1, 2)
    fig1.bar(x= [output["model"] for output in text_outputs], height=_)
    fig1.tick_params(axis='x', rotation=45)
    plt.sca(fig2)
    try:
        k = int(input("tell us how many clusters you want to form, where the number of clusters has to be at least 2, and at most the number of models you are testing: "))
    except ValueError:
        k = int(input("That was not an int, try again: "))
    result, labels = clustering.cluster_result(dists, k=k)
    fig2 = plt.scatter(result[:, 0], result[:, 1], c=labels)
    drift_results = get_semantic_drift(device=device, tst_model=model, given_tokenizer=tokenizer, outputs=text_outputs, ans=baseline_ans)
    drift_results["figure"].show()
    # anotherfig, axs = plt.subplots(1, len(drift_results["drifts"]), squeeze=False)
    # b: int = 0
    # for whatever in drift_results["drifts"]:
    #     plt.sca(axs[0, b])
    #     gramschmidt_process(whatever[2].cpu().numpy().tolist(), device=device)
    #     b+=1
    plt.show()
