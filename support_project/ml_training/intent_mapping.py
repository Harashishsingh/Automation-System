# ============================================================
# CATEGORY MAPPING
# ============================================================

CATEGORY_MAP = {
    "ACCOUNT": "account",
    "ORDER": "order",
    "REFUND": "refund",
    "INVOICE": "invoice",
    "CONTACT": "contact",
    "PAYMENT": "payment",
    "FEEDBACK": "feedback",
    "DELIVERY": "delivery",
    "SHIPPING": "shipping",
    "SUBSCRIPTION": "subscription",
    "CANCEL": "cancel",
}


# ============================================================
# HIGH PRIORITY INTENTS
# ============================================================

HIGH_PRIORITY_INTENTS = {
    "get_refund",
    "cancel_order",
    "payment_issue",
    "complaint",
    "delivery_problem",
    "check_cancellation_fee",
    "contact_human_agent",
}


# ============================================================
# MEDIUM PRIORITY INTENTS
# ============================================================

MEDIUM_PRIORITY_INTENTS = {
    "track_order",
    "change_order",
    "change_shipping_address",
    "edit_account",
    "track_refund",
    "recover_password",
    "registration_problems",
}


# ============================================================
# CATEGORY FUNCTION
# ============================================================

def get_category(raw_category):

    category = str(raw_category).upper().strip()

    return CATEGORY_MAP.get(category, "unknown")


# ============================================================
# PRIORITY FUNCTION
# ============================================================

def get_priority(intent):

    intent = str(intent).lower().strip()

    if intent in HIGH_PRIORITY_INTENTS:
        return "high"

    elif intent in MEDIUM_PRIORITY_INTENTS:
        return "medium"

    else:
        return "low"