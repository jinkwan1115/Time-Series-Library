import argparse
import pandas as pd
import torch
from chronos import ChronosPipeline, ChronosBoltPipeline

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="chronos", help="model for embedding (chronos or chronos_bolt)")
parser.add_argument("--context_len", type=int, default=192, help="context(input) length for embedding")
parser.add_argument("--data_path", type=str, default="/shared/timeSeries/forecasting/base/ETT-small/ETTh1.csv", help="data path for embedding")
parser.add_argument("--visualize", type=bool, default=False, help="visualize the embedding")

args = parser.parse_args()

df = pd.read_csv(args.data_path)
print(len(df["OT"]))
print(len(df["OT"][:args.context_len]))
# context must be either a 1D tensor, a list of 1D tensors, or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["OT"][:args.context_len])

# Visualization folder
output_dir = "visualization"
os.makedirs(output_dir, exist_ok=True)

if args.model == "chronos":
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    embeddings, _ = pipeline.embed(context)

    print(embeddings.size())
    print(embeddings)

if args.model == "chronos_bolt":
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    embeddings, _ = pipeline.embed(context)

    print(embeddings.size())
    print(embeddings)

if args.visualize:
    # tensor to numpy array
    embeddings_np = embeddings.squeeze(0).to(dtype=torch.float32).detach().cpu().numpy()

    # UMAP dimension reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings_np)

    # t-SNE 차원 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_embeddings = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(6, 6))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=10)
    plt.title("UMAP Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(os.path.join(output_dir, "umap_visualization.pdf"), format="pdf")
    plt.close()

    # t-SNE 시각화 저장
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=10)
    plt.title("t-SNE Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(os.path.join(output_dir, "tsne_visualization.pdf"), format="pdf")
    plt.close()