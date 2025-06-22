from django.db import models

# Create your models here.


class LoanRequest(models.Model):
    """A loan / credit request stored in the system.

    This model intentionally captures only a subset of fields for quick
    integration. It can be extended later to include all dataset features.
    """

    applicant_name = models.CharField(max_length=200)

    # Model output fields
    prediction = models.CharField(max_length=20, blank=True, null=True)
    probability = models.FloatField(null=True, blank=True)

    # New fields for full dataset features
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')], blank=True, null=True)
    married = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], blank=True, null=True)
    dependents = models.CharField(max_length=10, blank=True, null=True)
    education = models.CharField(max_length=20, choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')], blank=True, null=True)
    self_employed = models.CharField(max_length=10, choices=[('Yes', 'Yes'), ('No', 'No')], blank=True, null=True)
    applicant_income = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    coapplicant_income = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    loan_amount = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    loan_amount_term = models.IntegerField(blank=True, null=True)
    credit_history = models.FloatField(blank=True, null=True)
    property_area = models.CharField(max_length=20, choices=[('Urban', 'Urban'), ('Rural', 'Rural'), ('Semiurban', 'Semiurban')], blank=True, null=True)
    loan_status = models.CharField(max_length=10, choices=[('Y', 'Yes'), ('N', 'No')], blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.applicant_name} – {self.loan_amount} ({self.prediction})"
