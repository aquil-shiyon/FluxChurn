"""Data contract for FlixChurn.

Defines the canonical schemas at each layer of the data pipeline:
  raw schema → normalized schema → feature schema → model schema

This separation allows the underlying source dataset to change
without rewriting the entire ML system.

DATASET NOTE:
The source dataset (netflix_customer_churn_data.csv) is a flat snapshot
with 5,000 rows and 14 columns. It does NOT contain event-level timestamps.
The 'last_login_days' column serves as a recency proxy.
This is documented as a known limitation.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class SubscriptionType(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"


class Region(str, Enum):
    AFRICA = "Africa"
    ASIA = "Asia"
    EUROPE = "Europe"
    NORTH_AMERICA = "North America"
    SOUTH_AMERICA = "South America"
    OCEANIA = "Oceania"


class Device(str, Enum):
    TV = "TV"
    MOBILE = "Mobile"
    LAPTOP = "Laptop"
    DESKTOP = "Desktop"
    TABLET = "Tablet"


class PaymentMethod(str, Enum):
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    PAYPAL = "PayPal"
    GIFT_CARD = "Gift Card"
    CRYPTO = "Crypto"


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class FavoriteGenre(str, Enum):
    ACTION = "Action"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    DOCUMENTARY = "Documentary"
    HORROR = "Horror"
    ROMANCE = "Romance"
    SCI_FI = "Sci-Fi"


class RiskBand(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# --- Raw Schema ---
# Represents data exactly as it arrives from the source CSV.

class RawCustomerRecord(BaseModel):
    """Raw record from the source CSV. No transformations applied."""
    customer_id: str
    age: int
    gender: str
    subscription_type: str
    watch_hours: float
    last_login_days: int
    region: str
    device: str
    monthly_fee: float
    churned: int
    payment_method: str
    number_of_profiles: int
    avg_watch_time_per_day: float
    favorite_genre: str


# --- Normalized Schema ---
# Cleaned and validated version of raw data with proper types.

class NormalizedCustomerRecord(BaseModel):
    """Normalized customer record with validated types and enums."""
    customer_id: str
    age: int = Field(ge=18, le=120)
    gender: Gender
    subscription_type: SubscriptionType
    watch_hours: float = Field(ge=0)
    last_login_days: int = Field(ge=0)
    region: Region
    device: Device
    monthly_fee: float = Field(gt=0)
    churned: int = Field(ge=0, le=1)
    payment_method: PaymentMethod
    number_of_profiles: int = Field(ge=1, le=10)
    avg_watch_time_per_day: float = Field(ge=0)
    favorite_genre: FavoriteGenre

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("customer_id must not be empty")
        return v.strip()


# --- Feature Schema ---
# Model input features derived from normalized data.

class CustomerFeatures(BaseModel):
    """Feature vector for a single customer.

    These features are what the ML model consumes.
    Derived from normalized data via feature engineering.
    """
    customer_id: str

    # Numerical features (from raw data)
    age: float
    watch_hours: float
    last_login_days: float
    monthly_fee: float
    number_of_profiles: float
    avg_watch_time_per_day: float

    # Derived numerical features
    watch_hours_per_profile: float  # watch_hours / number_of_profiles
    fee_per_watch_hour: float  # monthly_fee / (watch_hours + 1) to avoid div-by-zero
    login_recency_score: float  # normalized recency indicator

    # Categorical features (encoded as strings, will be one-hot encoded)
    subscription_type: str
    region: str
    device: str
    payment_method: str
    gender: str
    favorite_genre: str


# --- Model Schema ---
# What the prediction output looks like.

class PredictionResult(BaseModel):
    """Output of a single customer churn prediction."""
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    prediction_horizon_days: int
    model_version: str
    prediction_timestamp: Optional[str] = None
    monthly_fee: float = 0.0
    mrr_at_risk: float = 0.0  # churn_probability * monthly_fee
    arr_at_risk: float = 0.0  # mrr_at_risk * 12
    churn_type_risk: str = "voluntary_engagement"  # "voluntary_engagement" or "involuntary_billing"


class ExplanationFeature(BaseModel):
    """A single feature's contribution to a prediction."""
    feature: str
    feature_value: float | str
    contribution: float  # SHAP value
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionExplanation(BaseModel):
    """SHAP-based explanation for a prediction."""
    customer_id: str
    model_version: str
    base_value: float  # Expected model output
    predicted_value: float
    top_contributors: list[ExplanationFeature]


class SimulationResult(BaseModel):
    """Result of a what-if simulation with financial ROI modeling."""
    customer_id: str
    baseline_probability: float
    scenario_probability: float
    probability_change_pp: float  # percentage points change
    risk_band_baseline: RiskBand
    risk_band_scenario: RiskBand
    model_version: str
    monthly_fee: float = 0.0
    mrr_saved: float = 0.0  # (baseline_probability - scenario_probability) * monthly_fee
    annual_value_preserved: float = 0.0  # mrr_saved * 12
    disclaimer: str = (
        "Scenario output represents model sensitivity and financial risk impact, "
        "not a verified causal estimate of customer intervention."
    )
