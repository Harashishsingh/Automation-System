from django.contrib import admin
from django.urls import path
from support_app import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.dashboard, name='dashboard'),
    path('ticket_list/', views.ticket_list, name='ticket_list'),
    path('tickets/new/', views.create_ticket, name='create_ticket'),
    path('tickets/generate/',views.generate_test_ticket,name='generate_test_ticket'),
    path('tickets/<int:pk>/', views.ticket_detail, name='ticket_detail'),
    path('automation-rules/', views.automation_rules_view, name='automation_rules'),
    path('decision-log/', views.decision_log_view, name='decision_log'),
    path('settings/', views.settings_view, name='settings'),
]