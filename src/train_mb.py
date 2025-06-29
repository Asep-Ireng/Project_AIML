import re
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- 1. Preprocessing Function ---
# Using Sastrawi for Indonesian stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text: str) -> str:
    """Cleans, lowercases, and stems the input text."""
    # Remove non-alphanumeric characters and lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    # Stem the text
    return stemmer.stem(text)

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 2. Load and Prepare Data ---
    print("Loading and preparing data...")
    # IMPORTANT: Create a 'dataset.csv' file in 'data/raw/'
    # It must have two columns: 'text' and 'label' (0 for Fakta, 1 for Hoax)
    try:
        df = pd.read_csv("../data/dataset.csv")
    except FileNotFoundError:
        print("Error: 'data/raw/dataset.csv' not found.")
        print("Please create the dataset file and run again.")
        exit()

    df["clean_text"] = df["text"].apply(preprocess)

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],  # Good for imbalanced datasets
    )
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing.")

    # --- 3. Create a Training Pipeline ---
    # This chains the vectorizer and the classifier together
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ]
    )

    # --- 4. Define Hyperparameter Grid for Tuning ---
    # We'll tune parameters for both the vectorizer and the classifier
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],  # Unigrams or Bigrams
        "tfidf__max_features": [3000, 5000],  # Max vocabulary size
        "clf__alpha": [0.1, 0.5, 1.0],  # Smoothing parameter for MNB
    }

    # --- 5. Perform Grid Search with Cross-Validation ---
    # This will find the best combination of parameters automatically
    print("\nStarting Grid Search for Multinomial Naive Bayes...")
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,  # 5-fold cross-validation
        scoring="f1_macro",  # Metric to optimize for
        n_jobs=-1,  # Use all available CPU cores
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    # --- 6. Evaluate the Best Model ---
    print("\n--- Evaluation Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Print the classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred, target_names=["Fakta", "Hoax"]))

    # --- 7. Save the Best Model ---
    # The saved model is the entire pipeline (vectorizer + classifier)
    with open("../models/classic_full/mnb_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nBest Multinomial Naive Bayes model saved to 'classic_full/mnb_model.pkl'")