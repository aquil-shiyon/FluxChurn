"""API integration and route tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "healthy"]
    assert "service" in data


def test_overview_endpoint():
    response = client.get("/api/v1/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers" in data
    assert "risk_distribution" in data
    assert "avg_churn_probability" in data
    assert "prediction_horizon_days" in data
    assert data["total_customers"] > 0


def test_risk_distribution_endpoint():
    response = client.get("/api/v1/risk-distribution")
    assert response.status_code == 200
    data = response.json()
    assert "total_eligible" in data
    assert "bands" in data
    assert "low" in data["bands"]
    assert "critical" in data["bands"]


def test_drivers_endpoint():
    response = client.get("/api/v1/drivers")
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) > 0
    assert "feature" in data["top_features"][0]
    assert "mean_abs_shap" in data["top_features"][0]


def test_cohorts_endpoint():
    response = client.get("/api/v1/cohorts?group_by=subscription_type")
    assert response.status_code == 200
    data = response.json()
    assert data["group_by"] == "subscription_type"
    assert "cohorts" in data
    assert "available_dimensions" in data


def test_customers_list_endpoint():
    response = client.get("/api/v1/customers?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "customers" in data
    assert "total" in data
    assert len(data["customers"]) <= 10
    if len(data["customers"]) > 0:
        first = data["customers"][0]
        assert "customer_id" in first
        assert "churn_probability" in first
        assert "risk_band" in first


def test_customer_risk_and_explanation():
    # First get a valid customer ID
    list_resp = client.get("/api/v1/customers?limit=1")
    assert list_resp.status_code == 200
    customers = list_resp.json().get("customers", [])
    if not customers:
        pytest.skip("No customers available in test dataset")

    cid = customers[0]["customer_id"]

    # Customer Risk
    risk_resp = client.get(f"/api/v1/customers/{cid}/risk")
    assert risk_resp.status_code == 200
    risk_data = risk_resp.json()
    assert risk_data["customer_id"] == cid
    assert "churn_probability" in risk_data
    assert "features" in risk_data

    # Customer Explanation
    exp_resp = client.get(f"/api/v1/customers/{cid}/explanation")
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert exp_data["customer_id"] == cid
    assert "top_contributors" in exp_data


def test_simulate_endpoint():
    list_resp = client.get("/api/v1/customers?limit=1")
    customers = list_resp.json().get("customers", [])
    if not customers:
        pytest.skip("No customers available for simulation")

    cid = customers[0]["customer_id"]

    payload = {
        "customer_id": cid,
        "scenario": {
            "watch_hours": 60,
            "last_login_days": 2,
        },
    }
    resp = client.post("/api/v1/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == cid
    assert "baseline_probability" in data
    assert "scenario_probability" in data
    assert "probability_change_pp" in data
    assert "risk_band_scenario" in data


def test_models_endpoints():
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "selected" in data


def test_monitoring_endpoints():
    # Data Quality
    dq = client.get("/api/v1/monitoring/data-quality")
    assert dq.status_code == 200
    assert "total_records" in dq.json()
    assert "schema_valid" in dq.json()

    # Drift
    drift = client.get("/api/v1/monitoring/drift")
    assert drift.status_code == 200
    assert "status" in drift.json()

    # Performance
    perf = client.get("/api/v1/monitoring/performance")
    assert perf.status_code == 200
