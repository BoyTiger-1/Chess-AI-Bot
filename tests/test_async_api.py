import pytest
from fastapi.testclient import TestClient
from ai_business_assistant.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

@patch("ai_business_assistant.api.auth.get_current_user")
@patch("ai_business_assistant.worker.celery_app.celery_app.send_task")
def test_create_forecast_async(mock_send_task, mock_get_current_user):
    mock_get_current_user.return_value = MagicMock(id=1)
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_send_task.return_value = mock_task
    
    response = client.post(
        "/api/v1/forecasts/create-async",
        json={"metric": "revenue", "periods": 12}
    )
    
    assert response.status_code == 200
    assert response.json()["task_id"] == "test-task-id"
    assert response.json()["status"] == "PENDING"

@patch("ai_business_assistant.api.auth.get_current_user")
@patch("ai_business_assistant.worker.task_handlers.get_task_status")
def test_get_task_status_endpoint(mock_get_task_status, mock_get_current_user):
    mock_get_current_user.return_value = MagicMock(id=1)
    mock_get_task_status.return_value = {"task_id": "test-id", "status": "SUCCESS", "result": {"data": 123}}
    
    response = client.get("/api/v1/tasks/test-id")
    
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
    assert response.json()["result"]["data"] == 123
