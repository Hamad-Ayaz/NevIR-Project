# Step 1: Install dependencies
!pip install -q transformers datasets scikit-learn

# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Imports
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 4: Load Dataset
train_dataset = load_dataset("orionweller/nevir", split="train")
test_dataset = load_dataset("orionweller/nevir", split="test")  # Use test set for final eval

# Step 5: Dataset Class
class NevIRCrossEncoderDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.examples = []
        for item in tqdm(dataset, desc="Preparing dataset"):
            q1, q2 = item['q1'], item['q2']
            d1, d2 = item['doc1'], item['doc2']
            self.examples += [
                {"query": q1, "doc": d1, "label": 1.0},
                {"query": q1, "doc": d2, "label": 0.0},
                {"query": q2, "doc": d1, "label": 0.0},
                {"query": q2, "doc": d2, "label": 1.0},
            ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["query"], ex["doc"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc = {k: v.squeeze() for k, v in enc.items()}
        enc["labels"] = torch.tensor(ex["label"], dtype=torch.float)
        return enc

# Step 6: Pairwise Evaluation Function
def evaluate_pairwise(model, tokenizer, dataset):
    model.eval()
    correct = 0
    total = len(dataset)
    with torch.no_grad():
        for ex in tqdm(dataset, desc="Evaluating pairwise"):
            q1, q2 = ex['q1'], ex['q2']
            d1, d2 = ex['doc1'], ex['doc2']

            def score(query, doc):
                enc = tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True).to(device)
                out = model(**enc).logits[0][0].item()
                return out

            s_q1_d1 = score(q1, d1)
            s_q1_d2 = score(q1, d2)
            s_q2_d1 = score(q2, d1)
            s_q2_d2 = score(q2, d2)

            if s_q1_d1 > s_q1_d2 and s_q2_d2 > s_q2_d1:
                correct += 1

    score = correct / total * 100
    print(f"\nFinal Negation-Aware Pairwise Score: {score:.2f}% ({correct}/{total})")
    return score

# Step 7: Train + Eval
def train_and_eval():
    model_name = "cross-encoder/qnli-electra-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    train_data = NevIRCrossEncoderDataset(train_dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Google Drive save path
    save_path = "/content/drive/MyDrive/nevir_model"

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0,
        logging_dir=f"{save_path}/logs",
        save_steps=1000,
        eval_strategy="no",
        report_to="none"
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = (preds > 0).astype(int)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model to Drive
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("\nFinal Evaluation on Test Set:")
    evaluate_pairwise(model, tokenizer, test_dataset)

# Run
train_and_eval()
