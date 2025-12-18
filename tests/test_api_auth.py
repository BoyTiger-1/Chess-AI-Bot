"""Unit tests for authentication service API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "auth_test.db"

    monkeypatch.setenv("POSTGRES_DSN", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "TEST_SECRET_KEY_CHANGE_ME_32_CHARS_MIN")

    from ai_business_assistant.shared.config import get_settings
    from ai_business_assistant.shared.db.session import get_engine, get_sessionmaker

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()

    from ai_business_assistant.auth_service.main import create_app

    app = create_app()

    with TestClient(app) as c:
        yield c


def test_register_user(client: TestClient):
    response = client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "roles" in data
    assert "user" in data["roles"]


def test_register_duplicate_email(client: TestClient):
    user_data = {
        "email": "duplicate@example.com",
        "password": "testpassword123",
    }

    assert client.post("/auth/register", json=user_data).status_code == 200

    response = client.post("/auth/register", json=user_data)
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"].lower()


def test_login(client: TestClient):
    client.post(
        "/auth/register",
        json={
            "email": "login@example.com",
            "password": "password12345",
        },
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "login@example.com",
            "password": "password12345",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(client: TestClient):
    response = client.post(
        "/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "wrongpassword",
        },
    )

    assert response.status_code == 401


def test_get_current_user(client: TestClient):
    client.post(
        "/auth/register",
        json={
            "email": "current@example.com",
            "password": "password12345",
        },
    )

    login = client.post(
        "/auth/login",
        json={
            "email": "current@example.com",
            "password": "password12345",
        },
    )

    token = login.json()["access_token"]

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "current@example.com"
