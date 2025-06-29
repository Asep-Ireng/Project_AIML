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

# Nama model dari Hugging Face Hub
MODEL_NAME = "indobenchmark/indobert-base-p1"

# --- 2. Custom Dataset Class ---
# Kita perlu membuat class ini agar data kita bisa dibaca oleh Trainer Hugging Face
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

        # Tokenisasi teks
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


# --- 3. Fungsi untuk Menghitung Metrik ---
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

    # Split data (kita butuh validation set untuk monitoring selama training)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )

    print(
        f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} testing."
    )

    # --- 5. Inisialisasi Tokenizer dan Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = HoaxDataset(
        texts=train_df.text.to_list(),
        labels=train_df.label.to_list(),
        tokenizer=tokenizer,
    )
    val_dataset = HoaxDataset(
        texts=val_df.text.to_list(),
        labels=val_df.label.to_list(),
        tokenizer=tokenizer,
    )

    # --- 6. Load Model ---
    # num_labels=2 karena kita punya 2 kelas: Fakta (0) dan Hoax (1)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)  # Pindahkan model ke GPU

    # --- 7. Tentukan Argumen Training ---
    training_args = TrainingArguments(
        output_dir="src/classic_full/classic_full/indobert_results",
        num_train_epochs=3,  # 3 epoch biasanya cukup untuk fine-tuning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluasi di setiap akhir epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # --- 8. Inisialisasi Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 9. Mulai Training ---
    print("\nStarting fine-tuning of IndoBERT...")
    trainer.train()

    # --- 10. Simpan Model Terbaik ---
    # Model terbaik akan otomatis di-load di akhir training
    best_model_path = "src/classic_full/classic_full/indobert_final"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest IndoBERT model saved to '{best_model_path}'")