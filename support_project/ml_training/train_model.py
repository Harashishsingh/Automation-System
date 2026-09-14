
"""
train_model.py

Training pipeline for the Bitext Customer Support project.

INPUT:
    cleaned_dataset.csv

OUTPUT:
    word_tfidf_vectorizer.pkl
    char_tfidf_vectorizer.pkl
    category_model.pkl
    intent_model.pkl
    priority_model.pkl
    model_metrics.pkl

All output files are saved inside the same ml_training folder.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import pandas as pd
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "cleaned_dataset.csv"
)


WORD_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "word_tfidf_vectorizer.pkl"
)

CHAR_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "char_tfidf_vectorizer.pkl"
)

CATEGORY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "category_model.pkl"
)

INTENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "intent_model.pkl"
)

PRIORITY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "priority_model.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "model_metrics.pkl"
)


# ============================================================
# STEP 1: LOAD CLEANED DATASET
# ============================================================

print("\n============================================")
print("LOADING CLEANED DATASET")
print("============================================")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\ncleaned_dataset.csv not found at:\n{DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)

print("Rows loaded:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# STEP 2: REQUIRED COLUMNS
# ============================================================

required_columns = [
    "message",
    "category",
    "intent",
    "priority"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# STEP 3: CLEAN BASIC DATA
# ============================================================

print("\n============================================")
print("DATA CHECK")
print("============================================")

df = df[required_columns].copy()

# Remove rows where message is missing
df = df.dropna(subset=["message"])

# Convert text to string
df["message"] = df["message"].astype(str).str.strip()

# Remove completely empty messages
df = df[df["message"] != ""]

# Convert labels to strings
df["category"] = (
    df["category"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["intent"] = (
    df["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["priority"] = (
    df["priority"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


# Remove rows with missing target labels
df = df[
    (df["category"] != "") &
    (df["intent"] != "") &
    (df["priority"] != "")
]


print("Rows after cleaning:", len(df))

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nPriority distribution:")
print(df["priority"].value_counts())


# ============================================================
# STEP 4: INPUT + TARGETS
# ============================================================

X = df["message"]

y_category = df["category"]

y_intent = df["intent"]

y_priority = df["priority"]


# ============================================================
# STEP 5: TRAIN / TEST SPLIT
# ============================================================

print("\n============================================")
print("TRAIN / TEST SPLIT")
print("============================================")

(
    X_train,
    X_test,
    y_category_train,
    y_category_test,
    y_intent_train,
    y_intent_test,
    y_priority_train,
    y_priority_test
) = train_test_split(
    X,
    y_category,
    y_intent,
    y_priority,
    test_size=0.20,
    random_state=42,
    stratify=y_category
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# STEP 6: WORD TF-IDF
# ============================================================

print("\n============================================")
print("WORD TF-IDF")
print("============================================")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=20000,
    sublinear_tf=True,
    stop_words="english"
)


X_train_word = word_vectorizer.fit_transform(
    X_train
)

X_test_word = word_vectorizer.transform(
    X_test
)


print(
    "Word features:",
    len(
        word_vectorizer.get_feature_names_out()
    )
)


# ============================================================
# STEP 7: CHARACTER TF-IDF
# ============================================================

print("\n============================================")
print("CHARACTER TF-IDF")
print("============================================")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)


X_train_char = char_vectorizer.fit_transform(
    X_train
)

X_test_char = char_vectorizer.transform(
    X_test
)


print(
    "Character features:",
    len(
        char_vectorizer.get_feature_names_out()
    )
)


# ============================================================
# STEP 8: COMBINE FEATURES
# ============================================================

print("\n============================================")
print("COMBINING TF-IDF FEATURES")
print("============================================")

X_train_vec = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()


X_test_vec = hstack(
    [
        X_test_word,
        X_test_char
    ]
).tocsr()


print(
    "Total features:",
    X_train_vec.shape[1]
)


# ============================================================
# STEP 9: MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n============================================")
    print(name)
    print("============================================")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Macro F1:",
        round(macro_f1, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# ============================================================
# STEP 10: BEST MODEL FUNCTION
# ============================================================

def choose_best(model_results):

    return max(
        model_results,
        key=lambda result: (
            result["macro_f1"],
            result["accuracy"]
        )
    )


# ============================================================
# STEP 11: CATEGORY MODELS
# ============================================================

print("\n\n============================================")
print("CATEGORY MODELS")
print("============================================")

category_results = []


category_lr = evaluate_model(

    "Logistic Regression - Category",

    LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ),

    X_train_vec,
    y_category_train,

    X_test_vec,
    y_category_test
)

category_results.append(
    category_lr
)


category_svm = evaluate_model(

    "Linear SVM - Category",

    LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    X_train_vec,
    y_category_train,

    X_test_vec,
    y_category_test
)

category_results.append(
    category_svm
)


best_category = choose_best(
    category_results
)

category_model = best_category["model"]


print("\n============================================")
print("BEST CATEGORY MODEL")
print("============================================")

print(
    "Selected:",
    type(category_model).__name__
)

print(
    "Accuracy:",
    round(
        best_category["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_category["macro_f1"],
        4
    )
)


# ============================================================
# STEP 12: INTENT MODELS
# ============================================================

print("\n\n============================================")
print("INTENT MODELS")
print("============================================")

intent_results = []


intent_lr = evaluate_model(

    "Logistic Regression - Intent",

    LogisticRegression(
        C=2.0,
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs"
    ),

    X_train_vec,
    y_intent_train,

    X_test_vec,
    y_intent_test
)

intent_results.append(
    intent_lr
)


intent_svm = evaluate_model(

    "Linear SVM - Intent",

    LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    X_train_vec,
    y_intent_train,

    X_test_vec,
    y_intent_test
)

intent_results.append(
    intent_svm
)


best_intent = choose_best(
    intent_results
)

intent_model = best_intent["model"]


print("\n============================================")
print("BEST INTENT MODEL")
print("============================================")

print(
    "Selected:",
    type(intent_model).__name__
)

print(
    "Accuracy:",
    round(
        best_intent["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_intent["macro_f1"],
        4
    )
)


# ============================================================
# STEP 13: PRIORITY MODELS
# ============================================================

print("\n\n============================================")
print("PRIORITY MODELS")
print("============================================")

priority_results = []


priority_lr = evaluate_model(

    "Logistic Regression - Priority",

    LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ),

    X_train_vec,
    y_priority_train,

    X_test_vec,
    y_priority_test
)

priority_results.append(
    priority_lr
)


priority_svm = evaluate_model(

    "Linear SVM - Priority",

    LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    X_train_vec,
    y_priority_train,

    X_test_vec,
    y_priority_test
)

priority_results.append(
    priority_svm
)


best_priority = choose_best(
    priority_results
)

priority_model = best_priority["model"]


print("\n============================================")
print("BEST PRIORITY MODEL")
print("============================================")

print(
    "Selected:",
    type(priority_model).__name__
)

print(
    "Accuracy:",
    round(
        best_priority["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_priority["macro_f1"],
        4
    )
)


# ============================================================
# STEP 14: SAVE VECTORIZERS
# ============================================================

print("\n============================================")
print("SAVING TF-IDF VECTORIZERS")
print("============================================")


joblib.dump(
    word_vectorizer,
    WORD_TFIDF_PATH
)


joblib.dump(
    char_vectorizer,
    CHAR_TFIDF_PATH
)


print(
    "Saved:",
    WORD_TFIDF_PATH
)

print(
    "Saved:",
    CHAR_TFIDF_PATH
)


# ============================================================
# STEP 15: SAVE MODELS
# ============================================================

print("\n============================================")
print("SAVING MODELS")
print("============================================")


joblib.dump(
    category_model,
    CATEGORY_MODEL_PATH
)


joblib.dump(
    intent_model,
    INTENT_MODEL_PATH
)


joblib.dump(
    priority_model,
    PRIORITY_MODEL_PATH
)


print(
    "Saved:",
    CATEGORY_MODEL_PATH
)

print(
    "Saved:",
    INTENT_MODEL_PATH
)

print(
    "Saved:",
    PRIORITY_MODEL_PATH
)


# ============================================================
# STEP 16: SAVE METRICS
# ============================================================

metrics = {

    "category": {
        "model": type(category_model).__name__,
        "accuracy": best_category["accuracy"],
        "macro_f1": best_category["macro_f1"]
    },

    "intent": {
        "model": type(intent_model).__name__,
        "accuracy": best_intent["accuracy"],
        "macro_f1": best_intent["macro_f1"]
    },

    "priority": {
        "model": type(priority_model).__name__,
        "accuracy": best_priority["accuracy"],
        "macro_f1": best_priority["macro_f1"]
    }
}


joblib.dump(
    metrics,
    METRICS_PATH
)


print(
    "Saved:",
    METRICS_PATH
)


# ============================================================
# STEP 17: REAL SAMPLE PREDICTION TEST
# ============================================================

print("\n============================================")
print("SAMPLE PREDICTION TEST")
print("============================================")


sample_texts = [

    "I want a refund for my order because I was charged for the wrong item.",

    "I cannot access my account and need help logging in.",

    "Where is my order? It has not arrived yet."

]


sample_word = word_vectorizer.transform(
    sample_texts
)

sample_char = char_vectorizer.transform(
    sample_texts
)


sample_features = hstack(
    [
        sample_word,
        sample_char
    ]
).tocsr()


sample_category_predictions = (
    category_model.predict(
        sample_features
    )
)


sample_intent_predictions = (
    intent_model.predict(
        sample_features
    )
)


sample_priority_predictions = (
    priority_model.predict(
        sample_features
    )
)


for i, text in enumerate(sample_texts):

    print("\nMessage:")
    print(text)

    print(
        "Category:",
        sample_category_predictions[i]
    )

    print(
        "Intent:",
        sample_intent_predictions[i]
    )

    print(
        "Priority:",
        sample_priority_predictions[i]
    )


# ============================================================
# STEP 18: FINAL SUMMARY
# ============================================================

print("\n\n============================================")
print("FINAL SUMMARY")
print("============================================")


print("\nCATEGORY")
print(
    "Model:",
    type(category_model).__name__
)

print(
    "Accuracy:",
    round(
        best_category["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_category["macro_f1"],
        4
    )
)


print("\nINTENT")
print(
    "Model:",
    type(intent_model).__name__
)

print(
    "Accuracy:",
    round(
        best_intent["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_intent["macro_f1"],
        4
    )
)


print("\nPRIORITY")
print(
    "Model:",
    type(priority_model).__name__
)

print(
    "Accuracy:",
    round(
        best_priority["accuracy"],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_priority["macro_f1"],
        4
    )
)


print("\n============================================")
print("FILES SAVED")
print("============================================")

print(
    "1.",
    WORD_TFIDF_PATH
)

print(
    "2.",
    CHAR_TFIDF_PATH
)

print(
    "3.",
    CATEGORY_MODEL_PATH
)

print(
    "4.",
    INTENT_MODEL_PATH
)

print(
    "5.",
    PRIORITY_MODEL_PATH
)

print(
    "6.",
    METRICS_PATH
)


print("\n============================================")
print("TRAINING COMPLETE")
print("============================================")

