import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# Prepare unique doc list for BM25
bm25_docs = []
doc_to_index = {}
for ex in dataset:
    for doc in [ex["doc1"], ex["doc2"]]:
        if doc not in doc_to_index:
            doc_to_index[doc] = len(bm25_docs)
            bm25_docs.append(doc)

tokenized_bm25 = [[word for word in doc.lower().split() if word not in ENGLISH_STOP_WORDS] for doc in bm25_docs]
bm25 = BM25Okapi(tokenized_bm25)

# Load stronger SBERT model
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# Load upgraded negation detection model
nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
nli_model.eval()

# Label indexes (RoBERTa-MNLI: contradiction = 0, entailment = 2)
entail_idx = 2
contradiction_idx = 0

def detect_negation(query):
    inputs = nli_tokenizer("This is true:", query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits[0].cpu().numpy())
    return probs[contradiction_idx] > probs[entail_idx]

def bm25_score(query, doc1, doc2, negated):
    tokens = [word for word in query.lower().split() if word not in ENGLISH_STOP_WORDS]
    scores = bm25.get_scores(tokens)
    score1 = scores[doc_to_index[doc1]]
    score2 = scores[doc_to_index[doc2]]
    return score2 > score1 if negated else score1 > score2

def sbert_score(query, doc1, doc2):
    embs = sbert.encode([query, doc1, doc2], convert_to_tensor=True)
    s1 = util.pytorch_cos_sim(embs[0], embs[1]).item()
    s2 = util.pytorch_cos_sim(embs[0], embs[2]).item()
    return s1 > s2

# Evaluation loop
correct_pairs = 0
for item in tqdm(dataset, desc="Evaluating Upgraded Hybrid Voting"):
    neg1 = detect_negation(item["q1"])
    neg2 = detect_negation(item["q2"])

    vote1 = [bm25_score(item["q1"], item["doc1"], item["doc2"], neg1), sbert_score(item["q1"], item["doc1"], item["doc2"])]
    vote2 = [bm25_score(item["q2"], item["doc1"], item["doc2"], neg2), sbert_score(item["q2"], item["doc1"], item["doc2"])]

    correct1 = sum(vote1) >= 1
    correct2 = sum(vote2) >= 1

    if correct1 and correct2:
        correct_pairs += 1

score = correct_pairs / len(dataset) * 100
print(f"\nFinal Score (BM25 + SBERT-MPNet + RoBERTa Negation): {score:.2f}%")
