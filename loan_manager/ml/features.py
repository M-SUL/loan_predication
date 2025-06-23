"""Feature engineering functions for loan prediction model."""

import pandas as pd


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to improve model performance."""
    X = X.copy()

    # Total income
    X["Total_Income"] = X["ApplicantIncome"] + X["CoapplicantIncome"]

    # Income to loan ratio
    X["Income_to_Loan_Ratio"] = X["Total_Income"] / X["LoanAmount"]

    # Monthly income
    X["Monthly_Income"] = X["Total_Income"] / 12

    # Monthly loan payment (assuming 8% annual interest)
    # Breaking the long calculation into parts
    monthly_rate = 0.08 / 12
    loan_term = X["Loan_Amount_Term"]
    compound_factor = (1 + monthly_rate) ** loan_term
    X["Monthly_Payment"] = (
        (X["LoanAmount"] * monthly_rate) * compound_factor / (compound_factor - 1)
    )

    # Payment to income ratio
    X["Payment_to_Income_Ratio"] = X["Monthly_Payment"] / X["Monthly_Income"]

    return X
