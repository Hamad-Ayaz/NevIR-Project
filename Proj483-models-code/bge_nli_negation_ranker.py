import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.special import softmax

# Load NevIR dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# Load sentence embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embedding_model.eval()

# Load LLM-based negation detector
nli_model_name = "typeform/distilbert-base-uncased-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_model.eval()

# Label indices for contradiction and entailment
entail_idx = 0  # 'entailment': 0, 'neutral': 1, 'contradiction': 2
contradiction_idx = 2

# Use GPU (cause i have an RTX2060)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = embedding_model.to(device)
nli_model = nli_model.to(device)

def detect_negation(query):
    """
    Returns True if the query contains negation or contradiction.
    """
    premise = "This is true:"
    inputs = nli_tokenizer(premise, query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits[0].cpu().numpy())
    return probs[contradiction_idx] > probs[entail_idx]

def rank_docs(query, doc1, doc2):
    """
    Ranks which doc is more relevant to the query using semantic similarity,
    with a negation aware flip if detected.
    """
    # Get embeddings
    embeddings = embedding_model.encode([query, doc1, doc2], convert_to_tensor=True)
    q_emb, d1_emb, d2_emb = embeddings[0], embeddings[1], embeddings[2]
    score1 = util.cos_sim(q_emb, d1_emb).item()
    score2 = util.cos_sim(q_emb, d2_emb).item()

    # Flip if negated
    if detect_negation(query):
        return score2 > score1
    else:
        return score1 > score2

# Evaluate on full dataset
correct_pairs = 0
for example in tqdm(dataset, desc="Evaluating"):
    q1_correct = rank_docs(example["q1"], example["doc1"], example["doc2"])
    q2_correct = rank_docs(example["q2"], example["doc1"], example["doc2"])
    if q1_correct and q2_correct:
        correct_pairs += 1

score = correct_pairs / len(dataset) * 100
print(f"\n✅ Final Pairwise Accuracy (BGE + LLM Negation): {score:.2f}%")
