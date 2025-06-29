import gradio as gr
import pickle
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. SETUP AWAL & PEMUATAN MODEL ---
# Bagian ini hanya akan berjalan sekali saat aplikasi pertama kali dijalankan.

print("Memuat semua model dan tokenizer...")

# Pastikan NLTK 'punkt' sudah diunduh untuk memecah kalimat
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Mengunduh NLTK 'punkt'...")
    nltk.download("punkt")

# Setup untuk Preprocessing model klasik
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def preprocess_classic(text: str) -> str:
    """Membersihkan dan melakukan stemming pada teks untuk model klasik."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return stemmer.stem(text)


# Muat semua model klasik yang sudah dilatih (dari dataset LENGKAP)
models = {}
try:
    # We will only load the best models from the FULL dataset for the app
    with open("models/classic_full/mnb_model.pkl", "rb") as f:
        models["Naive Bayes"] = pickle.load(f)
    with open("models/classic_full/lr_model.pkl", "rb") as f:
        models["Logistic Regression"] = pickle.load(f)
    with open("models/classic_full/svm_model.pkl", "rb") as f:
        models["Support Vector Machine"] = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Model .pkl tidak ditemukan. Pastikan file ada di folder 'models/classic_full/'. {e}")
    exit()


# Muat model dan tokenizer IndoBERT (dari dataset LENGKAP)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"IndoBERT akan berjalan di: {device}")
try:
    # The path should be a simple, direct path to the folder containing the model files.
    indobert_path = "models/indobert_full"
    indobert_tokenizer = AutoTokenizer.from_pretrained(indobert_path)
    indobert_model = AutoModelForSequenceClassification.from_pretrained(indobert_path)
    indobert_model.to(device)
    indobert_model.eval()
    models["IndoBERT"] = (indobert_model, indobert_tokenizer)
except OSError:
    print("Peringatan: Model IndoBERT tidak ditemukan di 'models/indobert_full/'. Opsi IndoBERT tidak akan tersedia.")

print("Semua model berhasil dimuat. Aplikasi siap.")


# --- 2. FUNGSI LOGIKA UTAMA ---
def analyze_text(model_choice, article_text):
    """
    Fungsi utama yang dipanggil oleh Gradio.
    Menganalisis teks input menggunakan model yang dipilih.
    """
    if not article_text.strip():
        return "Masukkan teks untuk dianalisis.", []

    # Pecah artikel menjadi kalimat-kalimat
    # sentences = nltk.sent_tokenize(article_text)

    #tanda baca parsing
    # sentences = [s.strip() for s in re.split(r'[.?!]', article_text) if s.strip()]

    #parsing tanda titik
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]


    highlight_data = []
    hoax_scores = []

    # Pilih model berdasarkan pilihan user
    if model_choice in ["Naive Bayes", "Logistic Regression", "Support Vector Machine"]:
        model = models[model_choice]
        for sentence in sentences:
            if len(sentence.strip()) < 5: continue  # Abaikan kalimat sangat pendek

            clean_sentence = preprocess_classic(sentence)

            # --- NEW: Check if the model can predict probabilities ---
            if hasattr(model, "predict_proba"):
                # For Naive Bayes and Logistic Regression
                prob_hoax = model.predict_proba([clean_sentence])[0][1]
            else:
                # For SVM, which gives a hard prediction (0 or 1)
                prediction = model.predict([clean_sentence])[0]
                prob_hoax = float(prediction)  # Convert 0 or 1 to a float

            hoax_scores.append(prob_hoax)
            label = "HOAX" if prob_hoax > 0.5 else "FAKTA"
            highlight_data.append((sentence, label))

    elif model_choice == "IndoBERT":
        if "IndoBERT" not in models:
            return "Error: Model IndoBERT tidak tersedia.", []
        model, tokenizer = models["IndoBERT"]
        for sentence in sentences:
            if len(sentence.strip()) < 5: continue

            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            prob_hoax = probs[0][1].item()  # Ambil probabilitas kelas 1 (Hoax)
            hoax_scores.append(prob_hoax)

            label = "HOAX" if prob_hoax > 0.5 else "FAKTA"
            highlight_data.append((sentence, label))

    # Buat ringkasan hasil
    if not hoax_scores:
        return "Tidak ada kalimat yang valid untuk dianalisis.", []

    avg_hoax_prob = sum(hoax_scores) / len(hoax_scores)
    hoax_sentence_count = sum(1 for score in hoax_scores if score > 0.5)

    summary_text = f"""
    ### Ringkasan Analisis
    - **Model yang Digunakan:** {model_choice}
    - **Probabilitas Hoax Rata-rata:** {avg_hoax_prob:.2%}
    - **Jumlah Kalimat Terdeteksi Hoax:** {hoax_sentence_count} dari {len(sentences)} kalimat.
    """

    return summary_text, highlight_data


# --- 3. DEFINISI UI GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Fact Checker") as demo:
    gr.Markdown("# AI Fact Checker - Analisis Teks Politik")
    gr.Markdown(
        "Masukkan judul, kalimat, atau artikel pendek untuk dianalisis. Model akan memberikan skor probabilitas hoax untuk setiap kalimat.")

    with gr.Row():
        # --- KOLOM KIRI: INPUT ---
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                lines=15,
                label="Teks Input",
                placeholder="Masukkan judul berita atau artikel pendek di sini..."
            )
            model_input = gr.Dropdown(
                choices=list(models.keys()),
                label="Pilih Model Analisis",
                value="Naive Bayes"  # Model default
            )
            btn_analyze = gr.Button("Analisis Teks", variant="primary")

        # --- KOLOM KANAN: OUTPUT ---
        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Ringkasan Hasil")
            highlight_output = gr.HighlightedText(
                label="Hasil Analisis per Kalimat",
                color_map={"HOAX": "red", "FAKTA": "green"},
                show_legend=True
            )

    # Hubungkan tombol dengan fungsi logika
    btn_analyze.click(
        fn=analyze_text,
        inputs=[model_input, text_input],
        outputs=[summary_output, highlight_output]
    )

# --- 4. LUNCURKAN APLIKASI ---
if __name__ == "__main__":
    demo.launch()