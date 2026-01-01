import pytest
from unittest.mock import MagicMock, patch
from ai_business_assistant.worker.tasks import generate_forecast, batch_market_analysis
from ai_business_assistant.worker.task_handlers import get_task_status

def test_generate_forecast_task():
    result = generate_forecast(10)
    assert result["periods"] == 10
    assert len(result["forecast"]) == 10

def test_batch_market_analysis_task():
    symbols = ["AAPL", "GOOGL"]
    result = batch_market_analysis(symbols)
    assert result["symbols"] == symbols
    assert "AAPL" in result["results"]
    assert "GOOGL" in result["results"]

@pytest.mark.asyncio
@patch("ai_business_assistant.worker.task_handlers.AsyncResult")
async def test_get_task_status_pending(mock_async_result):
    mock_result_instance = MagicMock()
    mock_result_instance.status = "PENDING"
    mock_result_instance.ready.return_value = False
    mock_async_result.return_value = mock_result_instance
    
    # Also mock DB check
    with patch("ai_business_assistant.worker.task_handlers.get_sessionmaker") as mock_session_maker:
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_session_maker.return_value.return_value.__aenter__.return_value = mock_session
        
        status = await get_task_status("fake-task-id")
        assert status["task_id"] == "fake-task-id"
        assert status["status"] == "PENDING"

@pytest.mark.asyncio
@patch("ai_business_assistant.worker.task_handlers.AsyncResult")
async def test_get_task_status_success(mock_async_result):
    mock_result_instance = MagicMock()
    mock_result_instance.status = "SUCCESS"
    mock_result_instance.ready.return_value = True
    mock_result_instance.successful.return_value = True
    mock_result_instance.result = {"foo": "bar"}
    mock_result_instance.name = "test_task"
    mock_async_result.return_value = mock_result_instance
    
    with patch("ai_business_assistant.worker.task_handlers.save_task_result") as mock_save:
        status = await get_task_status("fake-task-id")
        assert status["status"] == "SUCCESS"
        assert status["result"] == {"foo": "bar"}
        mock_save.assert_called_once()
