from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

# Load LLM for negation detection
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Function to detect negation in a query using the LLM
def is_negated_with_llm(query):
    prompt = (
        f"Determine whether the following question implies *negation* (e.g., denial, refusal, absence, or exclusion). "
        f"Respond with only 'yes' or 'no'.\n\n"
        f"Question: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    return "yes" in answer

# Load the NevIR dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# Prepare TF-IDF vectorizer
all_docs = [item["doc1"] for item in dataset] + [item["doc2"] for item in dataset]
vectorizer = TfidfVectorizer().fit(all_docs)

# Function to score document relevance with negation-aware flipping
def rank_docs(query, doc1, doc2):
    vecs = vectorizer.transform([query, doc1, doc2])
    q_vec = vecs[0]
    d1_vec = vecs[1]
    d2_vec = vecs[2]

    sim1 = cosine_similarity(q_vec, d1_vec)[0][0]
    sim2 = cosine_similarity(q_vec, d2_vec)[0][0]

    # Use LLM to detect negation
    if is_negated_with_llm(query):
        return sim2 > sim1
    else:
        return sim1 > sim2

# Evaluate pairwise accuracy
correct_pairs = 0
total = len(dataset)

for item in tqdm(dataset, desc="Evaluating"):
    correct_1 = rank_docs(item["q1"], item["doc1"], item["doc2"])
    correct_2 = rank_docs(item["q2"], item["doc1"], item["doc2"])
    if correct_1 and correct_2:
        correct_pairs += 1

accuracy = correct_pairs / total * 100
print(f"\nPairwise Accuracy (TF-IDF + LLM Negation Detection): {accuracy:.2f}%")
