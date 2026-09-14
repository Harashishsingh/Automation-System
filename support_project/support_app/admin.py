from django.contrib import admin

from .models import (
    AutomationRule,
    Ticket,
    DecisionLog,
    OrganizationSettings,
    WatchedInbox,
    NotificationPreference,
)


@admin.register(AutomationRule)
class AutomationRuleAdmin(admin.ModelAdmin):

    list_display = (
        'name',
        'category',
        'priority',
        'action',
        'is_active',
        'times_triggered',
        'created_at',
    )

    list_filter = (
        'is_active',
        'action',
        'category',
        'priority',
    )

    search_fields = (
        'name',
        'category',
        'priority',
    )


@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'title',
        'customer_name',
        'category',
        'priority',
        'status',
        'ai_predicted_action',
        'ai_confidence',
        'final_action',
        'was_overridden_by_human',
        'created_at',
    )

    list_filter = (
        'status',
        'final_action',
        'category',
        'priority',
        'was_overridden_by_human',
    )

    search_fields = (
        'title',
        'customer_name',
        'message',
    )

    readonly_fields = (
        'created_at',
    )


@admin.register(DecisionLog)
class DecisionLogAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'ticket',
        'category',
        'priority',
        'ml_predicted_action',
        'confidence',
        'safety_rule_applied',
        'final_action',
        'created_at',
    )

    list_filter = (
        'category',
        'priority',
        'ml_predicted_action',
        'final_action',
    )

    search_fields = (
        'ticket__title',
        'ticket__customer_name',
        'reason',
    )

    readonly_fields = (
        'ticket',
        'category',
        'priority',
        'ml_predicted_action',
        'confidence',
        'safety_rule_applied',
        'final_action',
        'reason',
        'created_at',
    )


@admin.register(OrganizationSettings)
class OrganizationSettingsAdmin(admin.ModelAdmin):

    list_display = (
        'business_name',
        'support_email',
        'timezone',
        'agent_is_active',
        'business_hours_start',
        'business_hours_end',
    )


@admin.register(WatchedInbox)
class WatchedInboxAdmin(admin.ModelAdmin):

    list_display = (
        'label',
        'email_address',
        'is_active',
        'connected_at',
    )

    list_filter = (
        'is_active',
    )

    search_fields = (
        'label',
        'email_address',
    )


@admin.register(NotificationPreference)
class NotificationPreferenceAdmin(admin.ModelAdmin):

    list_display = (
        'email_on_escalation',
        'email_on_new_ticket',
        'daily_summary_email',
        'summary_send_time',
    )

