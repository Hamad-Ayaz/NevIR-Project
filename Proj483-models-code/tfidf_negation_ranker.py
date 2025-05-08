from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 1. Load dataset
dataset = load_dataset("orionweller/NevIR", split="test")

# 2. Basic negation detector
negation_words = {"not", "no", "never", "n't", "none", "neither", "nobody", "nowhere", "nor"}

def has_negation(text):
    tokens = text.lower().split()
    return any(word in tokens for word in negation_words)

# 3. TF-IDF vectorizer
all_docs = [item["doc1"] for item in dataset] + [item["doc2"] for item in dataset]
vectorizer = TfidfVectorizer().fit(all_docs)

# 4. Score function with optional flip
def rank_docs(query, doc1, doc2):
    vecs = vectorizer.transform([query, doc1, doc2])
    q_vec = vecs[0]
    d1_vec = vecs[1]
    d2_vec = vecs[2]

    sim1 = cosine_similarity(q_vec, d1_vec)[0][0]
    sim2 = cosine_similarity(q_vec, d2_vec)[0][0]

    if has_negation(query):
        # Flip ranking if negated
        return sim2 > sim1
    else:
        return sim1 > sim2

# 5. Evaluate pairwise accuracy
correct_pairs = 0
total = len(dataset)

for item in tqdm(dataset, desc="Evaluating"):
    correct_1 = rank_docs(item["q1"], item["doc1"], item["doc2"])
    correct_2 = rank_docs(item["q2"], item["doc1"], item["doc2"])
    if correct_1 and correct_2:
        correct_pairs += 1

accuracy = correct_pairs / total * 100
print(f"\nPairwise Accuracy (TF-IDF + Negation Flip): {accuracy:.2f}%")