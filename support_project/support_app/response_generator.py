def generate_ai_response(ticket, category, intent, priority):
    """
    Generate a customer-facing response based on
    the ML classification.

    This is the response-generation layer.
    It does not decide whether the ticket should
    be resolved, drafted, or escalated.
    """

    category = (category or "").strip().lower()
    intent = (intent or "").strip().lower()
    priority = (priority or "").strip().lower()

    # --------------------------------------------------
    # REFUND
    # --------------------------------------------------

    if category == "refund":

        if "status" in intent or "pending" in intent:
            return (
                "We understand that you are checking the status of your "
                "refund. We are reviewing the refund details and will "
                "provide an update as soon as possible."
            )

        return (
            "We understand that you are requesting a refund. "
            "We will review your request and check the relevant order "
            "details to assist you with the refund process."
        )

    # --------------------------------------------------
    # INVOICE
    # --------------------------------------------------

    if category == "invoice":

        return (
            "We can help you with your invoice. Please provide the "
            "relevant order or account details so we can locate the "
            "invoice and assist you further."
        )

    # --------------------------------------------------
    # BILLING / PAYMENT
    # --------------------------------------------------

    if category == "billing":

        if "payment" in intent:
            return (
                "We understand that you are contacting us about a "
                "payment. We will review the payment details and "
                "help resolve the issue."
            )

        return (
            "We understand that you have a billing-related question. "
            "We will review your account and billing details and "
            "assist you further."
        )

    # --------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------

    if category == "account":

        if "password" in intent:
            return (
                "We can help you with your account password. "
                "Please follow the password reset process to securely "
                "update your password."
            )

        if "create" in intent or "register" in intent:
            return (
                "You can create an account by completing the account "
                "registration process. Please provide the required "
                "details and follow the instructions shown."
            )

        return (
            "We can help you with your account-related request. "
            "Please provide any relevant account details so we can "
            "assist you further."
        )

    # --------------------------------------------------
    # UNKNOWN / OTHER
    # --------------------------------------------------

    return (
        "Thank you for contacting support. We have received your "
        "request and will review it to determine the best way to "
        "assist you."
    )