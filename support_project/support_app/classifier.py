import os
import joblib


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_TRAINING_DIR = os.path.join(BASE_DIR, "ml_training")

WORD_TFIDF_PATH = os.path.join(
    ML_TRAINING_DIR,
    "word_tfidf_vectorizer.pkl"
)

CHAR_TFIDF_PATH = os.path.join(
    ML_TRAINING_DIR,
    "char_tfidf_vectorizer.pkl"
)

CATEGORY_MODEL_PATH = os.path.join(
    ML_TRAINING_DIR,
    "category_model.pkl"
)

INTENT_MODEL_PATH = os.path.join(
    ML_TRAINING_DIR,
    "intent_model.pkl"
)

PRIORITY_MODEL_PATH = os.path.join(
    ML_TRAINING_DIR,
    "priority_model.pkl"
)


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

REQUIRED_FILES = {
    "Word TF-IDF vectorizer": WORD_TFIDF_PATH,
    "Character TF-IDF vectorizer": CHAR_TFIDF_PATH,
    "Category model": CATEGORY_MODEL_PATH,
    "Intent model": INTENT_MODEL_PATH,
    "Priority model": PRIORITY_MODEL_PATH,
}

for file_name, file_path in REQUIRED_FILES.items():

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_name} not found at:\n{file_path}"
        )


# =========================================================
# LOAD VECTORIZERS
# =========================================================

word_tfidf = joblib.load(WORD_TFIDF_PATH)
char_tfidf = joblib.load(CHAR_TFIDF_PATH)


# =========================================================
# LOAD MODELS
# =========================================================

category_model = joblib.load(CATEGORY_MODEL_PATH)
intent_model = joblib.load(INTENT_MODEL_PATH)
priority_model = joblib.load(PRIORITY_MODEL_PATH)


# =========================================================
# HELPER: SVM CONFIDENCE
# =========================================================

def calculate_svm_confidence(model, features):
    """
    Calculate a relative confidence score from the SVM
    decision-function margin.

    LinearSVC does not produce probabilities.
    Therefore this is a bounded confidence estimate,
    not a true probability.
    """

    if not hasattr(model, "decision_function"):
        return 0.0

    scores = model.decision_function(features)

    # -----------------------------------------------------
    # Binary classification
    # -----------------------------------------------------

    if len(getattr(scores, "shape", ())) == 1:

        margin = abs(float(scores[0]))

    # -----------------------------------------------------
    # Multiclass classification
    # -----------------------------------------------------

    else:

        class_scores = scores[0]

        if len(class_scores) < 2:
            margin = abs(float(class_scores[0]))

        else:
            sorted_scores = sorted(
                class_scores,
                reverse=True
            )

            margin = (
                sorted_scores[0]
                - sorted_scores[1]
            )

    # -----------------------------------------------------
    # Convert margin into bounded 0-1 score
    # -----------------------------------------------------

    confidence = margin / (1.0 + margin)

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return confidence


# =========================================================
# CLASSIFY TICKET
# =========================================================

def classify_ticket(message):
    """
    Classify a customer-support ticket.

    Returns:
        category
        intent
        priority
        action
        category_confidence
        intent_confidence
        priority_confidence
        confidence
    """

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if message is None:
        raise ValueError(
            "Ticket message cannot be None."
        )

    message = str(message).strip()

    if not message:
        raise ValueError(
            "Ticket message cannot be empty."
        )


    # =====================================================
    # TF-IDF TRANSFORMATION
    # =====================================================

    word_features = word_tfidf.transform([message])
    char_features = char_tfidf.transform([message])


    # =====================================================
    # COMBINE WORD + CHARACTER FEATURES
    # =====================================================

    from scipy.sparse import hstack

    features = hstack(
        [
            word_features,
            char_features
        ]
    )


    # =====================================================
    # CATEGORY PREDICTION
    # =====================================================

    category_prediction = category_model.predict(
        features
    )[0]

    category = str(
        category_prediction
    ).strip().lower()


    # =====================================================
    # INTENT PREDICTION
    # =====================================================

    intent_prediction = intent_model.predict(
        features
    )[0]

    intent = str(
        intent_prediction
    ).strip().lower()


    # =====================================================
    # PRIORITY PREDICTION
    # =====================================================

    priority_prediction = priority_model.predict(
        features
    )[0]

    priority = str(
        priority_prediction
    ).strip().lower()


    # =====================================================
    # INDIVIDUAL CONFIDENCES
    # =====================================================

    category_confidence = calculate_svm_confidence(
        category_model,
        features
    )

    intent_confidence = calculate_svm_confidence(
        intent_model,
        features
    )

    priority_confidence = calculate_svm_confidence(
        priority_model,
        features
    )


    # =====================================================
    # FINAL CONFIDENCE
    # =====================================================
    #
    # Category is the main routing decision.
    # Intent and priority support that decision.
    #
    # Weighted combination:
    #
    # Category  = 50%
    # Intent    = 25%
    # Priority  = 25%
    #
    # This prevents one weak sub-model from completely
    # dominating the final confidence.
    # =====================================================

    confidence = (
        (category_confidence * 0.50)
        + (intent_confidence * 0.25)
        + (priority_confidence * 0.25)
    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )


    # =====================================================
    # ML ACTION
    # =====================================================
    #
    # IMPORTANT:
    # This is ONLY the ML recommendation.
    # engine.py will apply safety/automation rules
    # before deciding the final action.
    # =====================================================

    if confidence >= 0.80:

        action = "resolve"

    elif confidence >= 0.50:

        action = "draft"

    else:

        action = "escalate"


    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    print("\n========== ML PREDICTION ==========")
    print("Message:", message)
    print("Category:", category)
    print("Intent:", intent)
    print("Priority:", priority)
    print(
        "Category Confidence:",
        round(category_confidence, 4)
    )
    print(
        "Intent Confidence:",
        round(intent_confidence, 4)
    )
    print(
        "Priority Confidence:",
        round(priority_confidence, 4)
    )
    print(
        "Final Confidence:",
        round(confidence, 4)
    )
    print("Action:", action)
    print("===================================\n")


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {
        "category": category,
        "intent": intent,
        "priority": priority,
        "action": action,

        "category_confidence": round(
            category_confidence,
            4
        ),

        "intent_confidence": round(
            intent_confidence,
            4
        ),

        "priority_confidence": round(
            priority_confidence,
            4
        ),

        "confidence": round(
            confidence,
            4
        ),
    }

