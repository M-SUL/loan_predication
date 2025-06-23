import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for matplotlib

import pandas as pd
import numpy as np
import plotly.offline as opy
import plotly.graph_objs as go
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import LoanRequest
from .ml import model_service
from .ml.eda import (
    numerical_visualizations,
    categorical_visualizations,
    exclude_columns,
)


def request_list(request):
    """Render a table with saved loan requests."""
    requests_qs = LoanRequest.objects.order_by("-created_at")
    return render(
        request,
        "loan_manager/request_list.html",
        {"requests": requests_qs},
    )


def request_form(request):
    """Display/handle the add request form and perform model inference."""
    if request.method == "POST":
        applicant_name = request.POST.get("applicant")
        gender = request.POST.get("gender")
        married = request.POST.get("married")
        dependents = request.POST.get("dependents")
        education = request.POST.get("education")
        self_employed = request.POST.get("self_employed")
        applicant_income = request.POST.get("applicant_income")
        coapplicant_income = request.POST.get("coapplicant_income")
        loan_amount = request.POST.get("loan_amount")
        loan_amount_term = request.POST.get("loan_amount_term")
        credit_history = request.POST.get("credit_history")
        property_area = request.POST.get("property_area")
        loan_status = request.POST.get("loan_status")

        # Build feature dict for model
        features = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": (float(applicant_income) if applicant_income else 0.0),
            "CoapplicantIncome": (
                float(coapplicant_income) if coapplicant_income else 0.0
            ),
            "LoanAmount": float(loan_amount) if loan_amount else 0.0,
            "Loan_Amount_Term": (int(loan_amount_term) if loan_amount_term else 0),
            "Credit_History": (float(credit_history) if credit_history else 0.0),
            "Property_Area": property_area,
            # "Loan_Status": loan_status,  # Usually not used for prediction
        }
        try:
            prediction, prob = model_service.predict(features)
        except FileNotFoundError:
            prediction, prob = "Unknown", 0.0
            messages.warning(
                request,
                (
                    "Model not trained yet. "
                    "Please train the model to enable predictions."
                ),
            )

        LoanRequest.objects.create(
            applicant_name=applicant_name,
            gender=gender,
            married=married,
            dependents=dependents,
            education=education,
            self_employed=self_employed,
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_amount_term=loan_amount_term,
            credit_history=credit_history,
            property_area=property_area,
            loan_status=loan_status,
            prediction=prediction,
            probability=prob,
        )
        print(features)
        messages.success(
            request,
            f"Request saved with prediction: {prediction}",
        )
        return redirect("request_list")

    return render(request, "loan_manager/request_form.html")


def eda_view(request):
    # Load the dataset
    df = pd.read_csv("loan_manager/ml/loan_prediction.csv")
    df = exclude_columns(df)

    # Generate all visualizations
    numerical_visualizations(df)
    categorical_visualizations(df)

    # Prepare data for template
    context = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "desc_stats": df.describe().to_html(classes="table table-striped table-sm"),
        "skewness": {
            col: df[col].skew() for col in df.select_dtypes(include=[np.number]).columns
        },
        "categorical_cols": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
    }

    # Anomaly detection
    anomalies = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        anomalies[col] = {
            "count": len(outliers),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    context["anomalies"] = anomalies

    # Feature engineering suggestions
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    ratio_features = []
    interaction_features = []

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            ratio_features.append(f"{numerical_cols[i]} / {numerical_cols[j]}")
            interaction_features.append(f"{numerical_cols[i]} * {numerical_cols[j]}")

    context["ratio_features"] = ratio_features
    context["interaction_features"] = interaction_features

    # Data quality check
    constant_features = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_features.append(col)

    high_cardinality = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() > 20:
            high_cardinality.append((col, df[col].nunique()))

    context["constant_features"] = constant_features
    context["high_cardinality"] = high_cardinality

    return render(request, "loan_manager/eda.html", context)


def performance_view(request):
    """Display evaluation metrics and dummy confusion matrix/ROC curve."""
    metrics = model_service.load_metrics()
    # Dummy confusion matrix
    cm = go.Figure(
        data=go.Heatmap(
            z=metrics["confusion_matrix"],
            x=["Pred N", "Pred Y"],
            y=["True N", "True Y"],
        )
    )
    cm.update_layout(title="Confusion Matrix")
    cm_div = opy.plot(cm, auto_open=False, output_type="div")
    # Dummy ROC curve
    roc = go.Figure()
    roc.add_trace(
        go.Scatter(x=[0, 0.1, 0.2, 1], y=[0, 0.7, 0.9, 1], mode="lines", name="ROC")
    )
    roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )
    roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    roc_div = opy.plot(roc, auto_open=False, output_type="div")
    return render(
        request,
        "loan_manager/performance.html",
        {"metrics": metrics, "cm_div": cm_div, "roc_div": roc_div},
    )


def request_delete(request, pk):
    obj = get_object_or_404(LoanRequest, pk=pk)
    if request.method == "POST":
        obj.delete()
        messages.success(request, "Request deleted.")
        return redirect("request_list")
    return render(
        request,
        "loan_manager/request_confirm_delete.html",
        {"object": obj},
    )


def request_override(request, pk):
    obj = get_object_or_404(LoanRequest, pk=pk)
    if request.method == "POST":
        new_pred = request.POST.get("prediction")
        obj.prediction = new_pred
        obj.save()
        messages.success(request, f"Prediction overridden to: {new_pred}")
        return redirect("request_list")
    return render(
        request,
        "loan_manager/request_override.html",
        {"object": obj},
    )
