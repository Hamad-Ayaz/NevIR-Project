import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# TF-IDF setup
all_docs = [item["doc1"] for item in dataset] + [item["doc2"] for item in dataset]
vectorizer = TfidfVectorizer().fit(all_docs)

# Load SBERT
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Load Falcon-RW for negation prompting
llm_name = "tiiuae/falcon-rw-1b"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
llm_model.eval()

# Function to detect negation using LLM generation
def is_negated_with_llm(query):
    prompt = (
        f"Determine whether the following question implies *negation* "
        f"(e.g., denial, refusal, absence, or exclusion). "
        f"Respond with only 'yes' or 'no'.\n\n"
        f"Question: {query}\nAnswer:"
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=5)
    answer = llm_tokenizer.decode(output[0], skip_special_tokens=True).lower()
    return "yes" in answer

# TF-IDF ranking with negation flip
def tfidf_score(query, doc1, doc2, negated):
    vecs = vectorizer.transform([query, doc1, doc2])
    s1 = cosine_similarity(vecs[0], vecs[1])[0][0]
    s2 = cosine_similarity(vecs[0], vecs[2])[0][0]
    return s2 > s1 if negated else s1 > s2

# SBERT ranking
def sbert_score(query, doc1, doc2):
    embs = sbert.encode([query, doc1, doc2], convert_to_tensor=True)
    s1 = util.pytorch_cos_sim(embs[0], embs[1]).item()
    s2 = util.pytorch_cos_sim(embs[0], embs[2]).item()
    return s1 > s2

# Evaluation
correct_tfidf = 0
correct_sbert = 0
correct_hybrid = 0

for item in tqdm(dataset, desc="Evaluating Models"):
    q1, q2 = item["q1"], item["q2"]
    d1, d2 = item["doc1"], item["doc2"]

    # Negation detection via LLM
    neg1 = is_negated_with_llm(q1)
    neg2 = is_negated_with_llm(q2)

    # TF-IDF
    tf1 = tfidf_score(q1, d1, d2, neg1)
    tf2 = tfidf_score(q2, d1, d2, neg2)
    if tf1 and tf2:
        correct_tfidf += 1

    # SBERT
    sb1 = sbert_score(q1, d1, d2)
    sb2 = sbert_score(q2, d1, d2)
    if sb1 and sb2:
        correct_sbert += 1

    # Hybrid voting (majority)
    correct1 = sum([tf1, sb1]) >= 1
    correct2 = sum([tf2, sb2]) >= 1
    if correct1 and correct2:
        correct_hybrid += 1

# Results
total = len(dataset)
print(f"\nTF-IDF + FalconRW Negation Accuracy: {correct_tfidf / total * 100:.2f}%")
print(f"SBERT Accuracy: {correct_sbert / total * 100:.2f}%")
print(f"Hybrid Score (TF-IDF + SBERT + FalconRW Negation): {correct_hybrid / total * 100:.2f}%")
