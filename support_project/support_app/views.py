from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib import messages
import os
import random
import pandas as pd
from django.db.models import Q, Count, Case, When, IntegerField
from .models import (
    AutomationRule,
    Ticket,
    DecisionLog,
    OrganizationSettings,
    WatchedInbox,
    NotificationPreference,
)
from .engine import process_ticket


# ===================== TICKETS =====================

def ticket_list(request):
    tickets = Ticket.objects.all().order_by('-created_at')

    search_query = request.GET.get('search', '')
    if search_query:
        tickets = tickets.filter(
            Q(title__icontains=search_query) |
            Q(message__icontains=search_query) |
            Q(customer_name__icontains=search_query)
        )

    category_filter = request.GET.get('category', '')
    if category_filter:
        tickets = tickets.filter(category=category_filter)

    status_filter = request.GET.get('status', '')
    if status_filter:
        tickets = tickets.filter(status=status_filter)

    sort_filter = request.GET.get('sort', 'newest')
    if sort_filter == 'oldest':
        tickets = tickets.order_by('created_at')

    elif sort_filter == 'priority':
        tickets = tickets.annotate(
        priority_order=Case(
            When(priority='high', then=1),
            When(priority='medium', then=2),
            When(priority='low', then=3),
            default=4,
            output_field=IntegerField(),
        )
    ).order_by('priority_order', '-created_at')

    else:
        tickets = tickets.order_by('-created_at')

    all_tickets = Ticket.objects.all()

    # category_data replaces category_choices + category_counts.
    # Each item is a dict: {'category': 'refund', 'n': 3}
    # Template loops over this directly — no dict-lookup filter needed.
    category_data = (
        all_tickets.exclude(category='')
        .values('category')
        .annotate(n=Count('id'))
        .order_by('category')
    )

    context = {
        'tickets': tickets,
        'total_count': all_tickets.count(),
        'ai_resolved_count': all_tickets.filter(final_action='resolve').count(),
        'escalated_count': all_tickets.filter(final_action='escalate').count(),
        'pending_review_count': all_tickets.filter(status='open').count(),
        'search_query': search_query,
        'category_filter': category_filter,
        'status_filter': status_filter,
        'sort_filter': sort_filter,
        'category_data': category_data,
    }
    return render(request, 'ticket_list.html', context)


def ticket_detail(request, pk):
    ticket = get_object_or_404(Ticket, pk=pk)

    if request.method == 'POST':
        new_status = request.POST.get('status')
        if new_status:
            ticket.status = new_status
            ticket.was_overridden_by_human = True
            ticket.save()
        messages.success(request, "Ticket status updated.")
        return redirect('ticket_detail', pk=ticket.pk)

    context = {'ticket': ticket}
    return render(request, 'ticket_detail.html', context)


def create_ticket(request):
    if request.method == 'POST':
        ticket = Ticket.objects.create(
            title=request.POST.get('title', '').strip(),
            customer_name=request.POST.get('customer_name', '').strip(),
            message=request.POST.get('message', '').strip(),
        )
        result = process_ticket(ticket)
        messages.success(
            request,
            f"Ticket #{ticket.pk} classified as '{result['final_action']}' — {result['reason']}"
        )
        return redirect('ticket_list')

    return render(request, 'create_ticket.html')


# ===================== DASHBOARD =====================

def dashboard(request):
    all_tickets = Ticket.objects.all()
    context = {
        'total_tickets': all_tickets.count(),
        'open_tickets': all_tickets.filter(status='open').count(),
        'resolved_tickets': all_tickets.filter(final_action='resolve').count(),
        'escalated_tickets': all_tickets.filter(final_action='escalate').count(),
        'recent_tickets': all_tickets.order_by('-created_at')[:5],
    }
    return render(request, 'dashboard.html', context)


# ===================== AUTOMATION RULES (unchanged) =====================

def automation_rules_view(request):
    if request.method == 'POST':
        action = request.POST.get('form_action')

        if action == 'toggle':
            rule = get_object_or_404(AutomationRule, pk=request.POST.get('rule_id'))
            rule.is_active = not rule.is_active
            rule.save()

        elif action == 'create':
            AutomationRule.objects.create(
                name=request.POST.get('name', '').strip(),
                category=request.POST.get('category', '').strip(),
                priority=request.POST.get('priority', '').strip(),
                action=request.POST.get('action'),
            )

        elif action == 'delete':
            AutomationRule.objects.filter(pk=request.POST.get('rule_id')).delete()

        return redirect('automation_rules')

    rules = AutomationRule.objects.all().order_by('-created_at')

    context = {
        'rules': rules,
        'total_rules': rules.count(),
        'active_rules': rules.filter(is_active=True).count(),
        'paused_rules': rules.filter(is_active=False).count(),
        'auto_resolved_total': sum(
            r.times_triggered for r in rules if r.action == 'resolve'
        ),
    }
    return render(request, 'automation_rules.html', context)


# ===================== DECISION LOG =====================

# ===================== DECISION LOG =====================

def decision_log_view(request):
    logs = DecisionLog.objects.select_related(
        'ticket',
        'safety_rule_applied'
    ).all()

    context = {
        'logs': logs,
        'total_tickets': logs.count(),
        'escalated_count': logs.filter(
            final_action='escalate'
        ).count(),
        'resolved_count': logs.filter(
            final_action='resolve'
        ).count(),
        'draft_count': logs.filter(
            final_action='draft'
        ).count(),
        'overridden_count': Ticket.objects.filter(
            was_overridden_by_human=True
        ).count(),
    }

    return render(request, 'decision_log.html', context)




# ===================== SETTINGS (unchanged) =====================

def settings_view(request):
    org, _ = OrganizationSettings.objects.get_or_create(id=1)
    notif, _ = NotificationPreference.objects.get_or_create(id=1)
    inboxes = WatchedInbox.objects.all().order_by('-connected_at')

    if request.method == 'POST':
        action = request.POST.get('form_action')

        if action == 'update_general':
            org.business_name = request.POST.get('business_name')
            org.support_email = request.POST.get('support_email')
            org.timezone = request.POST.get('timezone')
            org.save()
            messages.success(request, "General settings updated.")

        elif action == 'toggle_agent':
            org.agent_is_active = not org.agent_is_active
            org.save()

        elif action == 'update_hours':
            org.business_hours_start = request.POST.get('business_hours_start') or None
            org.business_hours_end = request.POST.get('business_hours_end') or None
            org.save()
            messages.success(request, "Business hours updated.")

        elif action == 'add_inbox':
            WatchedInbox.objects.create(
                label=request.POST.get('label', ''),
                email_address=request.POST.get('email_address'),
            )

        elif action == 'toggle_inbox':
            inbox = get_object_or_404(WatchedInbox, pk=request.POST.get('inbox_id'))
            inbox.is_active = not inbox.is_active
            inbox.save()

        elif action == 'delete_inbox':
            WatchedInbox.objects.filter(pk=request.POST.get('inbox_id')).delete()

        elif action == 'update_notifications':
            notif.email_on_escalation = request.POST.get('email_on_escalation') == 'on'
            notif.email_on_new_ticket = request.POST.get('email_on_new_ticket') == 'on'
            notif.daily_summary_email = request.POST.get('daily_summary_email') == 'on'
            notif.save()
            messages.success(request, "Notification preferences updated.")

        elif action == 'change_password':
            form = PasswordChangeForm(user=request.user, data=request.POST)
            if form.is_valid():
                user = form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully.")
            else:
                messages.error(request, "Could not change password — check the form.")

        elif action == 'reset_automation_rules':
            AutomationRule.objects.all().delete()
            messages.success(request, "All automation rules were reset.")

        elif action == 'clear_all_tickets':
            Ticket.objects.all().delete()
            messages.success(request, "All tickets were cleared.")

        return redirect('settings')

    context = {
        'org': org,
        'notif': notif,
        'inboxes': inboxes,
        'active_inboxes': inboxes.filter(is_active=True).count(),
        'password_form': PasswordChangeForm(user=request.user) if request.user.is_authenticated else None,
    }
    return render(request, 'settings.html', context)

# Generate Ticket
def generate_test_ticket(request):
    if request.method == 'POST':

        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        dataset_path = os.path.join(
            base_dir,
            "ml_training",
            "cleaned_dataset.csv"
        )

        if not os.path.exists(dataset_path):
            messages.error(
                request,
                "cleaned_dataset.csv not found in ml_training folder."
            )
            return redirect('create_ticket')

        try:
            df = pd.read_csv(dataset_path)

            if "message" not in df.columns:
                messages.error(
                    request,
                    "Dataset does not contain a 'message' column."
                )
                return redirect('create_ticket')

            # Remove empty messages
            df = df[
                df["message"].notna()
                & (df["message"].astype(str).str.strip() != "")
            ]

            if df.empty:
                messages.error(
                    request,
                    "No valid messages found in dataset."
                )
                return redirect('create_ticket')

            # Pick one random customer message
            row = df.sample(n=1).iloc[0]

            message = str(row["message"]).strip()

            # Only the message is taken from dataset.
            # ML will independently predict category,
            # intent, priority and action.
            ticket = Ticket.objects.create(
                title="Automated Test Ticket",
                customer_name="Dataset Test Customer",
                message=message,
            )

            # Run complete AI + safety-rule pipeline
            result = process_ticket(ticket)

            messages.success(
                request,
                f"Automatic ticket #{ticket.pk} created — "
                f"{result['final_action']} — {result['reason']}"
            )

            return redirect('ticket_detail', pk=ticket.pk)

        except Exception as e:
            messages.error(
                request,
                f"Could not generate test ticket: {e}"
            )
            return redirect('create_ticket')

    return redirect('create_ticket')