"""Configuration loader for FlixChurn.

Loads YAML configuration files and environment variables.
All configurable parameters (prediction horizon, risk thresholds, etc.)
are centralized here to prevent hard-coding throughout the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: use working directory
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    filepath = CONFIGS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


# --- Pydantic config models ---

class RiskThresholds(BaseModel):
    """Risk band probability thresholds.

    Bands: [0, low) → Low, [low, moderate) → Moderate,
           [moderate, high) → High, [high, 1.0] → Critical
    """
    low: float = 0.3
    moderate: float = 0.5
    high: float = 0.7


class PredictionConfig(BaseModel):
    """Prediction parameters - loaded from configs/prediction.yaml."""
    prediction_horizon_days: int = 30
    observation_window_days: int = 60
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    min_tenure_days: int = 7
    production_model_version: str = "1.0.0"


class SplitConfig(BaseModel):
    """Train/validation/test split ratios."""
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2


class ModelParams(BaseModel):
    """Individual model parameters."""
    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    """Calibration configuration."""
    evaluate: bool = True
    method: str = "isotonic"
    apply_only_if_improved: bool = True


class SelectionCriteria(BaseModel):
    """Model selection criteria."""
    primary: str = "pr_auc"
    secondary: list[str] = Field(default_factory=lambda: ["recall_at_10", "brier_score"])
    minimum_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"pr_auc": 0.4, "roc_auc": 0.6}
    )


class ModelConfig(BaseModel):
    """Model training configuration - loaded from configs/model.yaml."""
    random_seed: int = 42
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: dict[str, ModelParams] = Field(default_factory=dict)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    selection_criteria: SelectionCriteria = Field(default_factory=SelectionCriteria)


class FeatureRangeConfig(BaseModel):
    """Valid range for a feature."""
    min: float
    max: float
    type: str = "continuous"


class FeatureConfig(BaseModel):
    """Feature configuration - loaded from configs/features.yaml."""
    feature_version: str = "1.0.0"
    numerical_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    simulatable_features: list[str] = Field(default_factory=list)
    feature_ranges: dict[str, FeatureRangeConfig] = Field(default_factory=dict)

    @property
    def all_feature_names(self) -> list[str]:
        """All model input features (numerical + categorical)."""
        return self.numerical_features + self.categorical_features


class MonitoringConfig(BaseModel):
    """Monitoring thresholds."""
    drift_threshold_psi: float = 0.2
    drift_threshold_ks: float = 0.1
    min_samples_drift: int = 100


class DatabaseConfig(BaseModel):
    """Database connection settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "flixchurn"
    user: str = "flixchurn"
    pool_size: int = 10
    max_overflow: int = 20


class AppSettings(BaseSettings):
    """Application settings from environment variables."""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "flixchurn"
    postgres_user: str = "flixchurn"
    postgres_password: str = "changeme_in_production"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "flixchurn-churn-prediction"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    app_env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    raw_data_dir: str = "data/raw"
    interim_data_dir: str = "data/interim"
    processed_data_dir: str = "data/processed"
    model_artifacts_dir: str = "data/models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# --- Singleton loaders ---

_prediction_config: PredictionConfig | None = None
_model_config: ModelConfig | None = None
_feature_config: FeatureConfig | None = None
_app_settings: AppSettings | None = None


def get_prediction_config() -> PredictionConfig:
    """Load and cache prediction configuration."""
    global _prediction_config
    if _prediction_config is None:
        raw = load_yaml("prediction.yaml")
        _prediction_config = PredictionConfig(**raw.get("prediction", raw))
    return _prediction_config


def get_model_config() -> ModelConfig:
    """Load and cache model configuration."""
    global _model_config
    if _model_config is None:
        raw = load_yaml("model.yaml")
        # Parse model params into ModelParams objects
        models = {}
        for key, val in raw.get("models", {}).items():
            models[key] = ModelParams(**val)
        raw["models"] = models
        _model_config = ModelConfig(**raw)
    return _model_config


def get_feature_config() -> FeatureConfig:
    """Load and cache feature configuration."""
    global _feature_config
    if _feature_config is None:
        raw = load_yaml("features.yaml")
        # Parse feature ranges
        ranges = {}
        for key, val in raw.get("feature_ranges", {}).items():
            ranges[key] = FeatureRangeConfig(**val)
        raw["feature_ranges"] = ranges
        _feature_config = FeatureConfig(**raw)
    return _feature_config


def get_app_settings() -> AppSettings:
    """Load and cache application settings."""
    global _app_settings
    if _app_settings is None:
        _app_settings = AppSettings()
    return _app_settings


def reset_configs() -> None:
    """Reset cached configs (useful for testing)."""
    global _prediction_config, _model_config, _feature_config, _app_settings
    _prediction_config = None
    _model_config = None
    _feature_config = None
    _app_settings = None
