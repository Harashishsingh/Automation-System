from .models import AutomationRule, DecisionLog
from .response_generator import generate_ai_response

def check_safety_rules(category, priority):
    """
    Check active automation/safety rules.

    A blank category or priority in a rule acts as a wildcard.
    Returns the first matching active rule.
    """

    rules = AutomationRule.objects.filter(is_active=True)

    # Normalize incoming values
    category = (category or "").strip().lower()
    priority = (priority or "").strip().lower()

    print("\n========== SAFETY RULE DEBUG ==========")
    print("Ticket category:", repr(category))
    print("Ticket priority:", repr(priority))
    print("Active rules:", rules.count())

    for rule in rules:

        rule_category = (rule.category or "").strip().lower()
        rule_priority = (rule.priority or "").strip().lower()

        print(
            "RULE:",
            rule.name,
            "| category =", repr(rule_category),
            "| priority =", repr(rule_priority),
            "| action =", rule.action,
            "| active =", rule.is_active
        )

        # Blank means "any"
        category_matches = (
            rule_category == "" or
            rule_category == category
        )

        priority_matches = (
            rule_priority == "" or
            rule_priority == priority
        )

        print(
            "   category_matches =", category_matches,
            "| priority_matches =", priority_matches
        )

        if category_matches and priority_matches:
            print(">>> MATCHED RULE:", rule.name)
            print("=======================================\n")
            return rule

    print(">>> NO SAFETY RULE MATCHED")
    print("=======================================\n")

    return None


def decide_final_action(category, priority, ml_action, confidence):
    """
    Decide the final action for a ticket.

    Priority:
    1. Active safety/automation rule
    2. ML confidence
    3. ML predicted action

    Returns a dictionary containing the complete decision.
    """

    safety_rule = check_safety_rules(category, priority)

    # -----------------------------------------------------
    # 1. SAFETY / AUTOMATION RULE
    # -----------------------------------------------------

    if safety_rule:

        final_action = safety_rule.action

        reason = (
            f"Safety rule applied: {safety_rule.name}"
        )

    # -----------------------------------------------------
    # 2. NO CONFIDENCE
    # -----------------------------------------------------

    elif confidence is None:

        final_action = "draft"

        reason = "No confidence available"

    # -----------------------------------------------------
    # 3. HIGH CONFIDENCE
    # -----------------------------------------------------

    elif confidence >= 0.90:

        final_action = ml_action

        reason = "High ML confidence"

    # -----------------------------------------------------
    # 4. MEDIUM CONFIDENCE
    # -----------------------------------------------------

    elif confidence >= 0.70:

        if ml_action == "escalate":
            final_action = "escalate"
        else:
            final_action = "draft"

        reason = "Medium ML confidence"

    # -----------------------------------------------------
    # 5. LOW CONFIDENCE
    # -----------------------------------------------------

    else:

        final_action = "draft"

        reason = "Low ML confidence"

    return {
        "category": category,
        "priority": priority,
        "ml_action": ml_action,
        "confidence": confidence,
        "final_action": final_action,
        "safety_rule": safety_rule,
        "reason": reason,
    }


def process_ticket(ticket):

    from .classifier import classify_ticket

    # -----------------------------------------------------
    # 1. ML CLASSIFICATION
    # -----------------------------------------------------

    prediction = classify_ticket(ticket.message)

    print("\n========== ML PREDICTION ==========")
    print("Message:", ticket.message)
    print("Category:", prediction.get("category"))
    print("Priority:", prediction.get("priority"))
    print("Action:", prediction.get("action"))
    print("Confidence:", prediction.get("confidence"))
    print("===================================\n")
    
        # -----------------------------------------------------
    # 1A. AI RESPONSE GENERATION
    # -----------------------------------------------------

    ai_response = generate_ai_response(
        ticket=ticket,
        category=prediction["category"],
        intent=prediction["intent"],
        priority=prediction["priority"],
    )

    # -----------------------------------------------------
    # 2. FINAL DECISION
    # -----------------------------------------------------

    result = decide_final_action(
        category=prediction["category"],
        priority=prediction["priority"],
        ml_action=prediction["action"],
        confidence=prediction["confidence"],
    )

    # -----------------------------------------------------
    # 3. SAVE ML + AUTOMATION RESULTS
    # -----------------------------------------------------

    ticket.category = result["category"]
    ticket.intent = prediction["intent"]

    ticket.priority = result["priority"]

    ticket.ai_predicted_action = result["ml_action"]
    ticket.ai_confidence = result["confidence"]

    ticket.final_action = result["final_action"]

    ticket.safety_rule_applied = result["safety_rule"]
    ticket.reason = result["reason"]
    ticket.ai_response = ai_response

    # -----------------------------------------------------
    # 3A. SYNC TICKET STATUS WITH FINAL ACTION
    # -----------------------------------------------------

    if result["final_action"] == "resolve":
        ticket.status = "resolved"

    elif result["final_action"] == "escalate":
        ticket.status = "escalated"

    else:
        ticket.status = "open"

    ticket.save()

    # -----------------------------------------------------
    # 3B. CREATE HISTORICAL DECISION LOG
    # -----------------------------------------------------

    DecisionLog.objects.create(
        ticket=ticket,
        category=result["category"],
        priority=result["priority"],
        ml_predicted_action=result["ml_action"],
        confidence=result["confidence"],
        safety_rule_applied=result["safety_rule"],
        final_action=result["final_action"],
        reason=result["reason"],
    )
    
    # -----------------------------------------------------
    # 4. UPDATE SAFETY RULE TRIGGER COUNT
    # -----------------------------------------------------

    if result["safety_rule"]:

        rule = result["safety_rule"]

        rule.times_triggered += 1
        rule.save(update_fields=["times_triggered"])

    # -----------------------------------------------------
    # 5. FINAL DEBUG OUTPUT
    # -----------------------------------------------------

    print("\n========== FINAL DECISION ==========")
    print("Category:", result["category"])
    print("Priority:", result["priority"])
    print("ML action:", result["ml_action"])
    print("Confidence:", result["confidence"])
    print(
        "Safety rule:",
        result["safety_rule"].name
        if result["safety_rule"]
        else "None"
    )
    print("Final action:", result["final_action"])
    print("Reason:", result["reason"])
    print("====================================\n")

    return result

