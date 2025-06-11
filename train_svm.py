import re
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- 1. Preprocessing Function (Identical) ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text: str) -> str:
    """Cleans, lowercases, and stems the input text."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return stemmer.stem(text)

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 2. Load and Prepare Data ---
    print("Loading and preparing data...")
    try:
        df = pd.read_csv("data/raw/dataset.csv")
    except FileNotFoundError:
        print("Error: 'data/raw/dataset.csv' not found.")
        print("Please create the dataset file and run again.")
        exit()

    df["clean_text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing.")

    # --- 3. Create a Training Pipeline ---
    # We are now using LinearSVC (Support Vector Classifier)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC(max_iter=2000, random_state=42)),
        ]
    )

    # --- 4. Define Hyperparameter Grid for Tuning ---
    # The main parameter for LinearSVC is 'C'
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 5000],
        "clf__C": [0.1, 1, 10],  # Regularization parameter
    }

    # --- 5. Perform Grid Search ---
    print("\nStarting Grid Search for Support Vector Machine (LinearSVC)...")
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    # --- 6. Evaluate the Best Model ---
    print("\n--- Evaluation Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred, target_names=["Fakta", "Hoax"]))

    # --- 7. Save the Best Model ---
    with open("models/svm_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nBest SVM model saved to 'models/svm_model.pkl'")