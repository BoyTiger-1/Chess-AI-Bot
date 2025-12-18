# API Documentation

## Base URL

```
Development: http://localhost:8000
Production: https://api.business-ai-assistant.com
```

## Authentication

All API endpoints (except `/auth/register` and `/auth/login`) require authentication using JWT Bearer tokens.

### Getting a Token

```bash
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=yourpassword
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

### Using the Token

Include the token in the Authorization header:

```bash
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## Rate Limiting

- **Per User**: 60 requests/minute, 1000 requests/hour
- **Unauthenticated**: 20 requests/minute

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

## Response Format

### Success Response

```json
{
  "data": {...},
  "message": "Success",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Response

```json
{
  "error": "ValidationError",
  "message": "Invalid input data",
  "details": [...],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Status Codes

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Endpoints

### Authentication

#### Register User

```http
POST /api/v1/auth/register
```

**Request Body**:
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response** (201):
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "is_verified": false,
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Login

```http
POST /api/v1/auth/login
```

**Request Body** (form-urlencoded):
```
username=user@example.com
password=SecurePass123!
```

**Response** (200):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

#### Get Current User

```http
GET /api/v1/auth/me
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "is_verified": true,
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Market Analysis

#### Analyze Market

```http
POST /api/v1/market/analyze
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "symbol": "AAPL",
  "market": "stock",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-12-31T23:59:59Z"
}
```

**Response** (200):
```json
{
  "symbol": "AAPL",
  "trend": "upward",
  "confidence": 0.85,
  "indicators": {
    "strength": 0.72,
    "change_points": 3
  },
  "volatility": 0.15,
  "recommendations": [
    "Monitor trend continuation",
    "Consider position adjustment based on volatility"
  ]
}
```

#### Get Sentiment Analysis

```http
GET /api/v1/market/sentiment/{symbol}
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "symbol": "AAPL",
  "sentiment_score": 0.72,
  "sentiment_label": "positive",
  "sources_analyzed": 50,
  "key_topics": ["earnings", "innovation", "market share"],
  "confidence": 0.85
}
```

### Financial Forecasting

#### Create Forecast

```http
POST /api/v1/forecasts/create
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "metric": "revenue",
  "periods": 12,
  "model_type": "prophet",
  "historical_data": [
    {"date": "2024-01-01", "value": 1000000},
    {"date": "2024-02-01", "value": 1050000}
  ]
}
```

**Response** (200):
```json
{
  "metric": "revenue",
  "model_type": "prophet",
  "forecast": [
    {"date": "2024-03-01", "value": 1100000},
    {"date": "2024-04-01", "value": 1150000}
  ],
  "confidence_intervals": [
    {"date": "2024-03-01", "lower": 1050000, "upper": 1150000},
    {"date": "2024-04-01", "lower": 1100000, "upper": 1200000}
  ],
  "accuracy_metrics": {
    "mape": 5.2,
    "rmse": 50000,
    "mae": 45000
  }
}
```

#### Create Scenario

```http
POST /api/v1/forecasts/scenarios
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "forecast_id": 1,
  "scenario_name": "Optimistic Growth",
  "assumptions": {
    "growth_rate": 0.15,
    "market_expansion": true,
    "new_products": 2
  }
}
```

### Competitive Intelligence

#### Add Competitor

```http
POST /api/v1/competitors/
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "name": "Competitor Inc",
  "website": "https://competitor.com",
  "industry": "Technology"
}
```

#### Analyze Competitor

```http
GET /api/v1/competitors/{competitor_id}/analysis
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "competitor_id": 1,
  "name": "Competitor Inc",
  "market_position": {
    "rank": 3,
    "market_share": 15.5
  },
  "strengths": [
    "Strong brand recognition",
    "Large customer base",
    "Efficient operations"
  ],
  "weaknesses": [
    "Limited innovation",
    "High pricing",
    "Slow market adaptation"
  ],
  "positioning": {
    "quality": "high",
    "price": "premium",
    "innovation": "medium"
  }
}
```

### Customer Analytics

#### Get Customer Segments

```http
GET /api/v1/customers/segments
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "segments": [
    {
      "id": 1,
      "name": "High Value",
      "customer_count": 150,
      "avg_lifetime_value": 5000,
      "characteristics": {
        "purchase_frequency": "high",
        "avg_order_value": 250
      }
    }
  ],
  "total_customers": 1000,
  "segmentation_quality": 0.75
}
```

#### Predict Churn

```http
GET /api/v1/customers/{customer_id}/churn
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "customer_id": 123,
  "churn_risk": 0.35,
  "risk_level": "medium",
  "key_factors": [
    "Low engagement score",
    "No purchases in 90 days",
    "Decreased login frequency"
  ],
  "recommendations": [
    "Send re-engagement campaign",
    "Offer personalized discount",
    "Reach out via preferred channel"
  ]
}
```

### Recommendations

#### Get Recommendations

```http
GET /api/v1/recommendations/?category=marketing&priority=high&limit=10
Authorization: Bearer {token}
```

**Response** (200):
```json
{
  "recommendations": [
    {
      "id": 1,
      "title": "Launch targeted email campaign",
      "description": "Engage high-value customers with personalized offers",
      "category": "marketing",
      "priority": "high",
      "confidence": 0.88,
      "explanation": "Based on customer segmentation and churn analysis...",
      "expected_impact": {
        "revenue_increase": 50000,
        "customer_retention": 0.15,
        "implementation_time": "2 weeks"
      }
    }
  ],
  "count": 5
}
```

#### Submit Feedback

```http
POST /api/v1/recommendations/feedback
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "recommendation_id": 1,
  "rating": 5,
  "implemented": true,
  "comments": "Campaign was successful, exceeded expectations"
}
```

### Data Management

#### Upload Data

```http
POST /api/v1/data/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data
```

**Form Data**:
```
file: [CSV/JSON/Excel file]
data_type: market|customer|competitor
```

**Response** (200):
```json
{
  "filename": "market_data_2024.csv",
  "size": 1048576,
  "data_type": "market",
  "status": "uploaded",
  "message": "File uploaded successfully. Processing will begin shortly."
}
```

### Data Export

#### Export as CSV

```http
GET /api/v1/export/csv/{data_type}?filters=...
Authorization: Bearer {token}
```

**Response**: CSV file download

#### Export as JSON

```http
GET /api/v1/export/json/{data_type}
Authorization: Bearer {token}
```

**Response**: JSON file download

### Webhooks

#### Create Webhook

```http
POST /api/v1/webhooks/
Authorization: Bearer {token}
```

**Request Body**:
```json
{
  "name": "Customer Churn Alert",
  "url": "https://your-app.com/webhooks/churn",
  "events": ["customer.churn_detected", "customer.segment_changed"],
  "secret": "webhook_secret_key"
}
```

**Response** (201):
```json
{
  "id": 1,
  "name": "Customer Churn Alert",
  "url": "https://your-app.com/webhooks/churn",
  "events": ["customer.churn_detected", "customer.segment_changed"],
  "is_active": true,
  "created_at": "2024-01-01T12:00:00Z"
}
```

## GraphQL

### Endpoint

```
POST /api/v1/graphql
Authorization: Bearer {token}
```

### Example Query

```graphql
query {
  forecasts(limit: 10) {
    id
    name
    metric
    model_type
    created_at
  }
}
```

### Example Mutation

```graphql
mutation {
  createForecast(metric: "revenue", periods: 12) {
    id
    name
    metric
    created_at
  }
}
```

## Pagination

List endpoints support pagination:

```http
GET /api/v1/forecasts/?limit=20&offset=0
```

**Response Headers**:
```
X-Total-Count: 100
X-Page: 1
X-Per-Page: 20
```

## Filtering and Sorting

```http
GET /api/v1/customers/?segment_id=1&sort=created_at&order=desc
```

## Error Codes

| Code | Description |
|------|-------------|
| `AUTH_001` | Invalid credentials |
| `AUTH_002` | Token expired |
| `AUTH_003` | Insufficient permissions |
| `VAL_001` | Validation error |
| `RATE_001` | Rate limit exceeded |
| `DATA_001` | Data not found |
| `DATA_002` | Data conflict |
| `SYS_001` | Internal server error |

## SDKs and Client Libraries

### Python

```python
from business_ai_client import BusinessAIClient

client = BusinessAIClient(
    api_key="your_api_key",
    base_url="https://api.business-ai-assistant.com"
)

# Get market sentiment
sentiment = client.market.get_sentiment("AAPL")
print(sentiment)

# Create forecast
forecast = client.forecasts.create(
    metric="revenue",
    periods=12,
    model_type="prophet"
)
```

### JavaScript/TypeScript

```typescript
import { BusinessAIClient } from '@business-ai/client';

const client = new BusinessAIClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.business-ai-assistant.com'
});

// Get recommendations
const recommendations = await client.recommendations.list({
  category: 'marketing',
  priority: 'high'
});
```

## Postman Collection

Download the Postman collection:
```
https://api.business-ai-assistant.com/postman/collection.json
```

## Support

- **Email**: api-support@business-ai-assistant.com
- **Documentation**: https://docs.business-ai-assistant.com
- **Status Page**: https://status.business-ai-assistant.com
