"""
GraphQL API endpoint using Strawberry.
"""

from typing import List, Optional
from datetime import datetime

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from ai_business_assistant.shared.logging import get_logger

logger = get_logger(__name__)


@strawberry.type
class User:
    id: int
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool


@strawberry.type
class Forecast:
    id: int
    name: str
    metric: str
    model_type: str
    created_at: datetime


@strawberry.type
class Recommendation:
    id: int
    title: str
    description: str
    category: str
    priority: str
    confidence: float


@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Welcome to Business AI Assistant GraphQL API"
    
    @strawberry.field
    async def user(self, id: int, info: Info) -> Optional[User]:
        logger.info(f"GraphQL query: user(id={id})")
        return None
    
    @strawberry.field
    async def forecasts(self, info: Info, limit: int = 10) -> List[Forecast]:
        logger.info(f"GraphQL query: forecasts(limit={limit})")
        return []
    
    @strawberry.field
    async def recommendations(self, info: Info, category: Optional[str] = None, limit: int = 10) -> List[Recommendation]:
        logger.info(f"GraphQL query: recommendations(category={category}, limit={limit})")
        return []


@strawberry.type
class Mutation:
    @strawberry.field
    async def create_forecast(self, metric: str, periods: int, info: Info) -> Forecast:
        logger.info(f"GraphQL mutation: createForecast(metric={metric}, periods={periods})")
        return Forecast(
            id=1,
            name=f"Forecast for {metric}",
            metric=metric,
            model_type="prophet",
            created_at=datetime.now()
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema, path="/")
