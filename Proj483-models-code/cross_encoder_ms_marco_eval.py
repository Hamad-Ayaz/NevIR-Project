import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from scipy.special import softmax

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# Load cross-encoder model
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

def score(query, doc):
    inputs = tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.item()  # Higher score = more relevant

# Evaluate pairwise accuracy
correct_pairs = 0
for item in tqdm(dataset, desc="Evaluating Cross-Encoder"):
    q1_score1 = score(item["q1"], item["doc1"])
    q1_score2 = score(item["q1"], item["doc2"])
    q2_score1 = score(item["q2"], item["doc1"])
    q2_score2 = score(item["q2"], item["doc2"])

    correct1 = q1_score1 > q1_score2
    correct2 = q2_score1 > q2_score2

    if correct1 and correct2:
        correct_pairs += 1

score_pct = correct_pairs / len(dataset) * 100
print(f"\nFinal Score (Cross-Encoder MS MARCO): {score_pct:.2f}%")
