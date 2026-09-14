import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("cleaned_dataset.csv")


print("\n============================================")
print("TRAIN / TEST LEAKAGE TEST")
print("============================================")


print("Total rows:", len(df))


# ============================================================
# SAME SPLIT AS YOUR TRAINING SCRIPT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)


print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english"
)


X_train = vectorizer.fit_transform(
    train_df["message"]
)


X_test = vectorizer.transform(
    test_df["message"]
)


# ============================================================
# SIMILARITY
# ============================================================

print("\nCalculating train/test similarities...")


similarities = cosine_similarity(
    X_test,
    X_train
)


# ============================================================
# FIND HIGHEST TRAIN SIMILARITY FOR EACH TEST MESSAGE
# ============================================================

max_similarity = similarities.max(
    axis=1
)


best_train_index = similarities.argmax(
    axis=1
)


test_df = test_df.copy()


test_df["max_train_similarity"] = max_similarity


test_df["best_train_index"] = best_train_index


# ============================================================
# THRESHOLDS
# ============================================================

thresholds = [
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]


print("\n============================================")
print("SIMILARITY RESULTS")
print("============================================")


for threshold in thresholds:

    count = (
        test_df["max_train_similarity"] >= threshold
    ).sum()

    percentage = (
        count / len(test_df)
    ) * 100

    print(
        f">= {threshold:.2f}: "
        f"{count} / {len(test_df)} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SHOW MOST SIMILAR EXAMPLES
# ============================================================

print("\n============================================")
print("TOP 30 TEST MESSAGES WITH TRAIN MATCHES")
print("============================================")


top_matches = test_df.sort_values(
    "max_train_similarity",
    ascending=False
).head(30)


for test_index, row in top_matches.iterrows():

    train_index = int(
        row["best_train_index"]
    )

    train_row = train_df.iloc[
        train_index
    ]


    print("\n--------------------------------------------")

    print(
        "Similarity:",
        round(
            row["max_train_similarity"],
            4
        )
    )

    print(
        "TEST message:",
        row["message"]
    )

    print(
        "TEST intent:",
        row["intent"]
    )

    print(
        "TRAIN message:",
        train_row["message"]
    )

    print(
        "TRAIN intent:",
        train_row["intent"]
    )


# ============================================================
# SAME INTENT VS DIFFERENT INTENT
# ============================================================

same_intent = 0
different_intent = 0


for test_index, row in test_df.iterrows():

    train_index = int(
        row["best_train_index"]
    )

    train_row = train_df.iloc[
        train_index
    ]

    if row["intent"] == train_row["intent"]:

        same_intent += 1

    else:

        different_intent += 1


print("\n============================================")
print("BEST MATCH LABEL ANALYSIS")
print("============================================")

print(
    "Same intent:",
    same_intent
)

print(
    "Different intent:",
    different_intent
)


# ============================================================
# SAVE RESULTS
# ============================================================

test_df.to_csv(
    "train_test_leakage_results.csv",
    index=False
)


print(
    "\nSaved:",
    "train_test_leakage_results.csv"
)