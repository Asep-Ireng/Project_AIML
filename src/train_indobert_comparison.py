import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: Running on CPU. This will be extremely slow.")
MODEL_NAME = "indobenchmark/indobert-base-p1"


# --- Dataset Class ---
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
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# --- Metrics Function ---
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    f1 = f1_score(p.label_ids, preds, average="macro")
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Data ---
    print("Loading and preparing data...")
    df = pd.read_csv("../data/dataset.csv")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    print(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} testing.")

    # --- Initialize ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = HoaxDataset(test_df.text.to_list(), test_df.label.to_list(), tokenizer)
    train_dataset = HoaxDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer)
    val_dataset = HoaxDataset(val_df.text.to_list(), val_df.label.to_list(), tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir="models/indobert_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics,
    )

    # --- Evaluate BEFORE Fine-Tuning ---
    print("\n--- Evaluating Model BEFORE Fine-Tuning (Zero-Shot Performance) ---")
    initial_results = trainer.evaluate(eval_dataset=test_dataset)

    # --- Train ---
    print("\n--- Starting Fine-Tuning of IndoBERT ---")
    trainer.train()

    # --- Generate Detailed Report for the FINAL model ---
    print("\n--- Generating Final Detailed Report on Test Set ---")
    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    print("\nClassification Report on Test Set (Fine-Tuned IndoBERT):")
    print(classification_report(y_true, y_pred, target_names=["Fakta", "Hoax"]))

    # # --- Save Model ---
    # best_model_path = "models/indobert_final"
    # trainer.save_model(best_model_path)
    # tokenizer.save_pretrained(best_model_path)
    # print(f"\nBest IndoBERT model saved to '{best_model_path}'")

    # --- Final Comparison Summary ---
    print("\n\n--- EXPERIMENT SUMMARY: BEFORE vs. AFTER ---")
    print(f"Model: {MODEL_NAME}")
    print("\nPerformance BEFORE Fine-Tuning:")
    print(f"  Accuracy: {initial_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"  F1 Score: {initial_results.get('eval_f1', 'N/A'):.4f}")

    # Get the final metrics from the detailed report for an accurate comparison
    final_f1_macro = f1_score(y_true, y_pred, average='macro')
    final_accuracy = accuracy_score(y_true, y_pred)

    print("\nPerformance AFTER Fine-Tuning:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  F1 Score: {final_f1_macro:.4f}")
    print("--------------------------------------------")