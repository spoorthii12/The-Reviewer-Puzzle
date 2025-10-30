"""
scibert_reviewer_topN.py

This script implements the SciBERT + Cosine similarity reviewer recommender with
Top-N per-author aggregation and outputs the top-k authors along with the
specific paper filenames that contributed to each author's score.

It is tailored to your dataset path:
/Users/spoorthivattem/Desktop/Dataset

Usage (example):
python scibert_reviewer_topN.py --dataset "/Users/spoorthivattem/Desktop/Dataset" \
    --query "/path/to/query.pdf" --topk 5 --topN 3 --cache scibert_cache.npz

Notes:
- Requires: torch, transformers, PyPDF2, numpy, scikit-learn, tqdm
- If you hit torch/safetensors issues, run with --use_safetensors or upgrade torch.

"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ---------------------------
# Utils: PDF -> text
# ---------------------------

def pdf_to_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""


# ---------------------------
# SciBERT embedding utilities
# ---------------------------

def load_scibert(model_name: str = "allenai/scibert_scivocab_uncased", use_safetensors: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        if use_safetensors:
            model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        else:
            model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"[WARN] initial model load failed ({e}). Retrying without use_safetensors...")
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def embed_text(text: str, tokenizer, model, device="cpu", max_length=512) -> np.ndarray:
    if not isinstance(text, str):
        text = str(text)
    if len(text.strip()) == 0:
        return None
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs, attention_mask)
    emb = pooled.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ---------------------------
# Dataset processing
# ---------------------------

def build_dataset_embeddings(dataset_dir: str,
                             tokenizer,
                             model,
                             device="cpu",
                             cache_path: str = "scibert_embeddings_cache.npz") -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    dataset_dir = Path(dataset_dir)
    author_to_papers = {}
    for author_folder in sorted(dataset_dir.iterdir()):
        if author_folder.is_dir():
            papers = []
            for p in sorted(author_folder.glob("*.pdf")):
                papers.append(str(p.resolve()))
            if papers:
                author_to_papers[author_folder.name] = papers

    # load cache if exists
    if os.path.exists(cache_path):
        try:
            print(f"[INFO] Loading embeddings cache from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            paper_embeddings = {k: cache[k].item() for k in cache.files}
            # Convert stored lists to arrays if necessary
            paper_embeddings = {p: (np.array(v) if not isinstance(v, np.ndarray) else v) for p, v in paper_embeddings.items()}
            return author_to_papers, paper_embeddings
        except Exception as e:
            print(f"[WARN] Failed loading cache ({e}), will recompute.")

    paper_embeddings = {}
    print("[INFO] Computing embeddings for dataset papers...")
    for author, papers in tqdm(author_to_papers.items()):
        for paper in papers:
            try:
                text = pdf_to_text(paper)
                if not text.strip():
                    print(f"[WARN] Empty text for {paper}, skipping.")
                    continue
                emb = embed_text(text, tokenizer, model, device=device)
                if emb is None:
                    continue
                paper_embeddings[paper] = emb
            except Exception as e:
                print(f"[WARN] Error embedding {paper}: {e}")

    # save cache as npz (store each paper embedding as array)
    try:
        np.savez_compressed(cache_path, **{p: paper_embeddings[p] for p in paper_embeddings})
        print(f"[INFO] Saved embeddings cache to {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache ({e})")

    return author_to_papers, paper_embeddings


# ---------------------------
# Top-N scoring
# ---------------------------

def compute_all_paper_similarities(query_emb: np.ndarray,
                                   author_to_papers: Dict[str, List[str]],
                                   paper_embeddings: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[str, float]]]:
    author_sims = {}
    for author, papers in author_to_papers.items():
        sims = []
        for p in papers:
            if p not in paper_embeddings:
                continue
            emb = paper_embeddings[p]
            sim = float(cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0, 0])
            sims.append((p, sim))
        if sims:
            author_sims[author] = sims
    return author_sims


def rank_authors_by_topN(author_sims: Dict[str, List[Tuple[str, float]]], topN: int = 3) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    author_scores = {}
    author_top_papers = {}
    for author, sims in author_sims.items():
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        top_n = sims_sorted[:min(topN, len(sims_sorted))]
        score = float(np.mean([s for (_, s) in top_n]))
        author_scores[author] = score
        author_top_papers[author] = top_n
    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    # return list of tuples (author, score, top_papers)
    return [(a, score, author_top_papers[a]) for (a, score) in ranked]


def find_top_k_authors_topN(query_path: str,
                            tokenizer,
                            model,
                            author_to_papers: Dict[str, List[str]],
                            paper_embeddings: Dict[str, np.ndarray],
                            device="cpu",
                            topk: int = 5,
                            topN: int = 3) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    text = pdf_to_text(query_path)
    if not text.strip():
        raise ValueError("Query PDF produced no text.")
    q_emb = embed_text(text, tokenizer, model, device=device)

    author_sims = compute_all_paper_similarities(q_emb, author_to_papers, paper_embeddings)
    ranked = rank_authors_by_topN(author_sims, topN=topN)
    return ranked[:topk]


# ---------------------------
# CLI / main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, default="/Users/spoorthivattem/Desktop/Dataset", help="Path to dataset directory (one subfolder per author).")
    parser.add_argument("--query", required=True, help="Path to query PDF (test paper).")
    parser.add_argument("--topk", type=int, default=5, help="Return top-k authors.")
    parser.add_argument("--topN", type=int, default=3, help="Number of top papers per author to average.")
    parser.add_argument("--cache", default="scibert_embeddings_cache.npz", help="Cache path for paper embeddings.")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--use_safetensors", action="store_true", help="Try loading model using safetensors (if available).")
    args = parser.parse_args()

    print("[INFO] Loading SciBERT model...")
    tokenizer, model = load_scibert(use_safetensors=args.use_safetensors)
    device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)

    author_to_papers, paper_embeddings = build_dataset_embeddings(args.dataset, tokenizer, model, device=device, cache_path=args.cache)

    if not paper_embeddings:
        print("[ERROR] No paper embeddings computed. Exiting.")
        return

    print(f"[INFO] Querying {args.query} ...")
    topk_authors = find_top_k_authors_topN(
        args.query,
        tokenizer,
        model,
        author_to_papers,
        paper_embeddings,
        device=device,
        topk=args.topk,
        topN=args.topN
    )

    print("\nTop-k authors (Top-N paper aggregation):")
    for i, (author, score, top_papers) in enumerate(topk_authors, 1):
        print(f"{i}. {author:30s}  score: {score:.4f}")
        for (paper_path, sim) in top_papers:
            fname = os.path.basename(paper_path)
            print(f"    - {fname:50s} sim={sim:.4f}")


if __name__ == "__main__":
    main()
