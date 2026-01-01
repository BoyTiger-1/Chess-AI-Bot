from __future__ import annotations

import hashlib
import time
from typing import Any, List, Dict

from ai_business_assistant.worker.celery_app import celery_app
from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)

def _fake_embedding(text: str, *, dim: int = 64) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dim * 4 // len(digest)) + 1))[: dim * 4]

    floats: list[float] = []
    for i in range(0, len(raw), 4):
        val = int.from_bytes(raw[i : i + 4], "big")
        floats.append((val % 10_000) / 10_000.0)
    return floats


@celery_app.task(name="ai_business_assistant.worker.tasks.embed_message")
def embed_message(*, message_id: str, content: str) -> dict[str, Any]:
    # Placeholder for actual embedding logic
    time.sleep(1)  # Simulate work
    dim = 64
    return {"message_id": message_id, "embedding_dim": dim}


@celery_app.task(name="ai_business_assistant.worker.tasks.batch_market_analysis")
def batch_market_analysis(symbols: List[str], webhook_url: str = None) -> Dict[str, Any]:
    logger.info(f"Starting batch market analysis for {symbols}")
    results = {}
    for symbol in symbols:
        time.sleep(2)  # Simulate analysis for each symbol
        results[symbol] = {
            "sentiment": 0.75,
            "trend": "bullish",
            "volatility": "low"
        }
    logger.info(f"Completed batch market analysis for {symbols}")
    return {"symbols": symbols, "results": results}


@celery_app.task(name="ai_business_assistant.worker.tasks.generate_forecast")
def generate_forecast(periods: int, webhook_url: str = None) -> Dict[str, Any]:
    logger.info(f"Starting forecast generation for {periods} periods")
    time.sleep(min(periods * 0.1, 30))  # Simulate long-running job
    forecast_data = [i * 1.5 for i in range(periods)]
    logger.info(f"Completed forecast generation for {periods} periods")
    return {"periods": periods, "forecast": forecast_data}


@celery_app.task(name="ai_business_assistant.worker.tasks.segment_customers")
def segment_customers(dataset_id: str, webhook_url: str = None) -> Dict[str, Any]:
    logger.info(f"Starting customer segmentation for dataset {dataset_id}")
    time.sleep(10)  # Simulate heavy processing
    segments = {
        "high_value": ["user1", "user2"],
        "at_risk": ["user3"],
        "new": ["user4", "user5"]
    }
    logger.info(f"Completed customer segmentation for dataset {dataset_id}")
    return {"dataset_id": dataset_id, "segments": segments}


@celery_app.task(name="ai_business_assistant.worker.tasks.competitive_analysis")
def competitive_analysis(competitors: List[str], webhook_url: str = None) -> Dict[str, Any]:
    logger.info(f"Starting competitive analysis for {competitors}")
    time.sleep(5)  # Simulate analysis
    analysis = {comp: {"market_share": "10%", "ranking": i+1} for i, comp in enumerate(competitors)}
    logger.info(f"Completed competitive analysis for {competitors}")
    return {"competitors": competitors, "analysis": analysis}


@celery_app.task(name="ai_business_assistant.worker.tasks.generate_report")
def generate_report(report_type: str, parameters: Dict[str, Any], webhook_url: str = None) -> Dict[str, Any]:
    logger.info(f"Starting report generation: {report_type}")
    time.sleep(7)  # Simulate report generation
    report_url = f"https://storage.example.com/reports/{report_type}_{int(time.time())}.pdf"
    logger.info(f"Completed report generation: {report_type}")
    return {"report_type": report_type, "report_url": report_url}


from celery.signals import task_postrun
from ai_business_assistant.worker.task_handlers import save_task_result, notify_webhook

@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **signal_kwargs):
    """Signal fired after a task finishes."""
    import asyncio
    
    logger.info(f"Task {task_id} finished with state {state}")
    
    webhook_url = kwargs.get("webhook_url")
    
    # We need to run the async handlers in a synchronous signal handler
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Save to DB
        loop.run_until_complete(save_task_result(
            task_id=task_id,
            task_name=task.name,
            status=state,
            result=retval if state == "SUCCESS" else None,
            error=str(retval) if state == "FAILURE" else None,
            webhook_url=webhook_url
        ))
        
        # Notify Webhook if present
        if webhook_url:
            payload = {
                "task_id": task_id,
                "task_name": task.name,
                "status": state,
                "result": retval if state == "SUCCESS" else None,
                "error": str(retval) if state == "FAILURE" else None
            }
            loop.run_until_complete(notify_webhook(webhook_url, payload))
            
    finally:
        loop.close()
