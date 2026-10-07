"""Data validation module for FlixChurn.

Implements explicit validation for:
- Required columns
- Data types
- Missing values
- Duplicates
- Impossible numerical values
- Categorical value validity
- Customer ID validity

Validation failures are collected and reported, not silently repaired.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ml.data_contract import (
    Device,
    FavoriteGenre,
    Gender,
    PaymentMethod,
    Region,
    SubscriptionType,
)
from ml.ingestion import EXPECTED_RAW_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: str  # "error", "warning", "info"
    check: str  # Name of the check
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    dataset_name: str
    total_records: int
    total_columns: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status}: {self.total_records} records, "
            f"{self.total_columns} columns, "
            f"{self.error_count} errors, {self.warning_count} warnings"
        )


def validate_raw_data(df: pd.DataFrame, dataset_name: str = "raw") -> ValidationReport:
    """Run all validation checks on raw data.

    Args:
        df: Raw DataFrame to validate.
        dataset_name: Name for the validation report.

    Returns:
        ValidationReport with all findings.
    """
    report = ValidationReport(
        dataset_name=dataset_name,
        total_records=len(df),
        total_columns=len(df.columns),
    )

    _check_required_columns(df, report)
    _check_missing_values(df, report)
    _check_duplicates(df, report)
    _check_customer_id_validity(df, report)
    _check_numerical_ranges(df, report)
    _check_categorical_values(df, report)
    _check_target_variable(df, report)

    logger.info(report.summary())
    for issue in report.issues:
        log_fn = logger.error if issue.severity == "error" else logger.warning
        log_fn(f"[{issue.check}] {issue.message}")

    return report


def _check_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify all expected columns are present."""
    missing = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_RAW_COLUMNS)

    if missing:
        report.issues.append(ValidationIssue(
            severity="error",
            check="required_columns",
            message=f"Missing columns: {sorted(missing)}",
            details={"missing_columns": sorted(missing)},
        ))
    if extra:
        report.issues.append(ValidationIssue(
            severity="info",
            check="extra_columns",
            message=f"Extra columns found (will be ignored): {sorted(extra)}",
            details={"extra_columns": sorted(extra)},
        ))


def _check_missing_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for missing values in each column."""
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            severity = "error" if pct > 10 else "warning"
            report.issues.append(ValidationIssue(
                severity=severity,
                check="missing_values",
                message=f"Column '{col}' has {count} missing values ({pct:.1f}%)",
                details={"column": col, "count": int(count), "percentage": pct},
            ))
    else:
        report.issues.append(ValidationIssue(
            severity="info",
            check="missing_values",
            message="No missing values detected",
        ))


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for duplicate customer IDs and fully duplicate rows."""
    if "customer_id" in df.columns:
        dup_ids = df["customer_id"].duplicated().sum()
        if dup_ids > 0:
            report.issues.append(ValidationIssue(
                severity="error",
                check="duplicate_customer_ids",
                message=f"Found {dup_ids} duplicate customer_id values",
                details={"count": int(dup_ids)},
            ))

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        report.issues.append(ValidationIssue(
            severity="warning",
            check="duplicate_rows",
            message=f"Found {dup_rows} fully duplicate rows",
            details={"count": int(dup_rows)},
        ))


def _check_customer_id_validity(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify customer IDs are non-empty and well-formed."""
    if "customer_id" not in df.columns:
        return

    empty_ids = df["customer_id"].apply(
        lambda x: pd.isna(x) or (isinstance(x, str) and not x.strip())
    ).sum()
    if empty_ids > 0:
        report.issues.append(ValidationIssue(
            severity="error",
            check="customer_id_validity",
            message=f"Found {empty_ids} empty/null customer_id values",
            details={"count": int(empty_ids)},
        ))


def _check_numerical_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check numerical columns for impossible values."""
    range_checks = {
        "age": (18, 120, "Age"),
        "watch_hours": (0, 500, "Watch hours"),
        "last_login_days": (0, 365, "Last login days"),
        "monthly_fee": (0.01, 100, "Monthly fee"),
        "number_of_profiles": (1, 10, "Number of profiles"),
        "avg_watch_time_per_day": (0, 24, "Avg watch time per day"),
    }

    for col, (min_val, max_val, label) in range_checks.items():
        if col not in df.columns:
            continue

        below = (df[col] < min_val).sum()
        above = (df[col] > max_val).sum()

        if below > 0:
            report.issues.append(ValidationIssue(
                severity="error",
                check="numerical_range",
                message=f"{label}: {below} values below minimum ({min_val})",
                details={"column": col, "below_min": int(below)},
            ))
        if above > 0:
            # Watch time per day can exceed 24 if it's actually cumulative
            severity = "warning" if col == "avg_watch_time_per_day" else "error"
            report.issues.append(ValidationIssue(
                severity=severity,
                check="numerical_range",
                message=f"{label}: {above} values above maximum ({max_val})",
                details={"column": col, "above_max": int(above)},
            ))


def _check_categorical_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify categorical columns contain only expected values."""
    category_checks = {
        "subscription_type": {e.value for e in SubscriptionType},
        "region": {e.value for e in Region},
        "device": {e.value for e in Device},
        "payment_method": {e.value for e in PaymentMethod},
        "gender": {e.value for e in Gender},
        "favorite_genre": {e.value for e in FavoriteGenre},
    }

    for col, valid_values in category_checks.items():
        if col not in df.columns:
            continue

        actual_values = set(df[col].dropna().unique())
        invalid = actual_values - valid_values

        if invalid:
            report.issues.append(ValidationIssue(
                severity="warning",
                check="categorical_values",
                message=f"Column '{col}' has unexpected values: {sorted(invalid)}",
                details={"column": col, "invalid_values": sorted(invalid)},
            ))


def _check_target_variable(df: pd.DataFrame, report: ValidationReport) -> None:
    """Validate the churn target variable."""
    if "churned" not in df.columns:
        report.issues.append(ValidationIssue(
            severity="error",
            check="target_variable",
            message="Target column 'churned' not found",
        ))
        return

    unique_vals = set(df["churned"].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        report.issues.append(ValidationIssue(
            severity="error",
            check="target_variable",
            message=f"Target 'churned' contains values other than 0/1: {unique_vals}",
        ))

    churn_rate = df["churned"].mean()
    report.issues.append(ValidationIssue(
        severity="info",
        check="class_distribution",
        message=f"Churn rate: {churn_rate:.1%} ({df['churned'].sum()} churned / {len(df)} total)",
        details={"churn_rate": churn_rate, "churned_count": int(df["churned"].sum())},
    ))


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """Generate a data quality summary suitable for the monitoring API."""
    report = validate_raw_data(df, "quality_check")
    missing_dict = {str(col): int(df[col].isna().sum()) for col in df.columns}
    dupes = int(df.duplicated(subset=["customer_id"]).sum()) if "customer_id" in df.columns else int(df.duplicated().sum())

    return {
        "total_records": report.total_records,
        "total_columns": report.total_columns,
        "passed": report.passed,
        "schema_valid": report.passed,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "missing_values": missing_dict,
        "duplicate_count": dupes,
        "issues": [issue.message for issue in report.issues if issue.severity == "error"],
        "checks": [
            {
                "severity": issue.severity,
                "check": issue.check,
                "message": issue.message,
                "details": issue.details,
            }
            for issue in report.issues
        ],
    }
