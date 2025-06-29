import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

# --- 1. Konfigurasi dan Pengecekan GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: Running on CPU. This will be extremely slow.")

MODEL_NAME = "indobenchmark/indobert-base-p1"

# --- 2. Custom Dataset Class (Identical) ---
class HoaxDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# --- 3. Fungsi untuk Menghitung Metrik (Identical) ---
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    f1 = f1_score(p.label_ids, preds, average="macro")
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 4. Load Data ---
    print("Loading and preparing data...")
    try:
        df = pd.read_csv("../data/dataset.csv")
    except FileNotFoundError:
        print("Error: 'data/raw/dataset.csv' not found.")
        exit()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    print(
        f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} testing."
    )

    # --- 5. Inisialisasi Tokenizer dan Semua Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = HoaxDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer)
    val_dataset = HoaxDataset(val_df.text.to_list(), val_df.label.to_list(), tokenizer)
    # **NEW**: Create the test dataset object for evaluation
    test_dataset = HoaxDataset(test_df.text.to_list(), test_df.label.to_list(), tokenizer)

    # --- 6. Load Pre-trained Model ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # --- 7. Tentukan Argumen Training ---
    training_args = TrainingArguments(
        output_dir="../models/indobert_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1 score to find the best model
    )

    # --- 8. Inisialisasi Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 9. **NEW**: Evaluate BEFORE Fine-Tuning ---
    print("\n--- Evaluating Model BEFORE Fine-Tuning (Zero-Shot Performance) ---")
    initial_results = trainer.evaluate(eval_dataset=test_dataset)
    print(initial_results)

    # --- 10. Mulai Training ---
    print("\n--- Starting Fine-Tuning of IndoBERT ---")
    trainer.train()

    # --- 11. **NEW**: Evaluate AFTER Fine-Tuning ---
    print("\n--- Evaluating Model AFTER Fine-Tuning ---")
    finetuned_results = trainer.evaluate(eval_dataset=test_dataset)
    print(finetuned_results)

    # --- 12. Simpan Model Terbaik ---
    best_model_path = "../models/indobert_full/indobert_final"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest IndoBERT model saved to '{best_model_path}'")

    # --- 13. **NEW**: Final Comparison Summary ---
    print("\n\n--- EXPERIMENT SUMMARY ---")
    print(f"Model: {MODEL_NAME}")
    print("\nPerformance BEFORE Fine-Tuning:")
    print(f"  Accuracy: {initial_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  F1 Score: {initial_results.get('eval_f1', 'N/A'):.4f}")
    print("\nPerformance AFTER Fine-Tuning:")
    print(f"  Accuracy: {finetuned_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  F1 Score: {finetuned_results.get('eval_f1', 'N/A'):.4f}")
    print("--------------------------")