from django.db import models


class AutomationRule(models.Model):
    """
    Handwritten safety rules.
    These are NOT the main ML logic — they are safety guardrails
    that override ML decisions when matched.
    """
    name = models.CharField(max_length=120)
    category = models.CharField(max_length=30, blank=True)   # blank = "any"
    priority = models.CharField(max_length=10, blank=True)   # blank = "any"
    action = models.CharField(max_length=20, choices=[
        ('resolve', 'Auto-resolve'),
        ('escalate', 'Escalate'),
        ('draft', 'Draft only'),
    ])
    is_active = models.BooleanField(default=True)
    times_triggered = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "Automation_table"

    def __str__(self):
        return self.name


class Ticket(models.Model):
    title = models.CharField(max_length=200)
    customer_name = models.CharField(max_length=120)
    message = models.TextField()
    category = models.CharField(max_length=30, blank=True)
    intent = models.CharField(max_length=100, blank=True)
    priority = models.CharField(max_length=10, blank=True)
    status = models.CharField(
    max_length=20,
    choices=[
        ('open', 'Open'),
        ('progress', 'In progress'),
        ('resolved', 'Resolved'),
        ('escalated', 'Escalated'),
    ],
    default='open'
)
    created_at = models.DateTimeField(auto_now_add=True)

    # ML + engine output
    ai_predicted_action = models.CharField(max_length=20, blank=True)
    ai_confidence = models.FloatField(null=True, blank=True)
    final_action = models.CharField(max_length=20, blank=True)
    safety_rule_applied = models.ForeignKey(
        AutomationRule, null=True, blank=True, on_delete=models.SET_NULL
    )
    was_overridden_by_human = models.BooleanField(default=False)
    reason = models.CharField(max_length=200, blank=True)
    ai_response = models.TextField(blank=True)

    def __str__(self):
        return self.title
    
class DecisionLog(models.Model):
    """
    Historical record of every AI + automation decision.

    This is separate from Ticket so that the decision history
    remains preserved even if the ticket is later modified.
    """

    ticket = models.ForeignKey(
        Ticket,
        on_delete=models.CASCADE,
        related_name="decision_logs"
    )

    category = models.CharField(max_length=30, blank=True)
    intent = models.CharField(max_length=100, blank=True)
    priority = models.CharField(max_length=10, blank=True)

    ml_predicted_action = models.CharField(max_length=20, blank=True)

    confidence = models.FloatField(
        null=True,
        blank=True
    )

    safety_rule_applied = models.ForeignKey(
        AutomationRule,
        null=True,
        blank=True,
        on_delete=models.SET_NULL
    )

    final_action = models.CharField(max_length=20, blank=True)

    reason = models.CharField(max_length=200, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Decision for Ticket #{self.ticket_id}"


class OrganizationSettings(models.Model):
    """Singleton — general identity + agent master switch + business hours."""
    business_name = models.CharField(max_length=120, default="My Company")
    support_email = models.EmailField(blank=True)
    timezone = models.CharField(max_length=50, default="UTC")
    agent_is_active = models.BooleanField(default=True)
    business_hours_start = models.TimeField(null=True, blank=True)
    business_hours_end = models.TimeField(null=True, blank=True)

    class Meta:
        db_table = "org_settings_table"


class WatchedInbox(models.Model):
    """One row per inbox the AI agent reads tickets from."""
    label = models.CharField(max_length=60, blank=True)
    email_address = models.EmailField()
    is_active = models.BooleanField(default=True)
    connected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "watched_inbox_table"

    def __str__(self):
        return self.label or self.email_address


class NotificationPreference(models.Model):
    """Singleton — when the admin gets emailed by the system itself."""
    email_on_escalation = models.BooleanField(default=True)
    email_on_new_ticket = models.BooleanField(default=False)
    daily_summary_email = models.BooleanField(default=True)
    summary_send_time = models.TimeField(null=True, blank=True)

    class Meta:
        db_table = "notification_pref_table"