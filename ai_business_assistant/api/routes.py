"""
API route aggregation - imports all routers for main app.
"""

from ai_business_assistant.api.auth import router as auth_router
from ai_business_assistant.api.market import router as market_router
from ai_business_assistant.api.forecasting import router as forecasting_router
from ai_business_assistant.api.competitive import router as competitive_router
from ai_business_assistant.api.customer import router as customer_router
from ai_business_assistant.api.recommendations import router as recommendations_router
from ai_business_assistant.api.data import router as data_router
from ai_business_assistant.api.export import router as export_router
from ai_business_assistant.api.webhooks import router as webhooks_router
from ai_business_assistant.api.tasks import router as tasks_router
from ai_business_assistant.api.models import router as models_router

__all__ = [
    "auth_router",
    "market_router",
    "forecasting_router",
    "competitive_router",
    "customer_router",
    "recommendations_router",
    "data_router",
    "export_router",
    "webhooks_router",
    "tasks_router",
    "models_router",
]
