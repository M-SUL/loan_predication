from django.contrib import admin
from .models import LoanRequest


@admin.register(LoanRequest)
class LoanRequestAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "applicant_name",
        "loan_amount",
        "prediction",
        "created_at",
    )
    list_filter = ("prediction",)
    search_fields = ("applicant_name",)
