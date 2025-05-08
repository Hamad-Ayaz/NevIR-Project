import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
test_dataset = load_dataset("orionweller/nevir", split="test")

# Load model and tokenizer
model_name = "cross-encoder/qnli-electra-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# Pairwise accuracy function
def evaluate_pairwise(model, tokenizer, dataset):
    correct = 0
    total = len(dataset)
    with torch.no_grad():
        for ex in tqdm(dataset, desc="Evaluating baseline"):
            q1, q2 = ex['q1'], ex['q2']
            d1, d2 = ex['doc1'], ex['doc2']

            def score(query, doc):
                inputs = tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True).to(device)
                out = model(**inputs).logits[0][0].item()
                return out

            s_q1_d1 = score(q1, d1)
            s_q1_d2 = score(q1, d2)
            s_q2_d1 = score(q2, d1)
            s_q2_d2 = score(q2, d2)

            if s_q1_d1 > s_q1_d2 and s_q2_d2 > s_q2_d1:
                correct += 1

    score = correct / total * 100
    print(f"\nBaseline Pairwise Score (no fine-tuning): {score:.2f}% ({correct}/{total})")
    return score

# Run evaluation
evaluate_pairwise(model, tokenizer, test_dataset)
