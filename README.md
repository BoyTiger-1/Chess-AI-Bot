# Business AI Assistant

A complete, production-ready **AI-powered Business Intelligence Platform** providing market analysis, financial forecasting, competitive intelligence, customer behavior analytics, and strategic recommendations.

## 🚀 Features

### Backend Infrastructure
- **FastAPI** with async support, comprehensive middleware, and error handling
- **PostgreSQL** database with SQLAlchemy ORM and async support
- **Redis** caching layer for high-performance data access
- **JWT Authentication** with OAuth2 provider integration support
- **Role-Based Access Control (RBAC)** with permissions system
- **Rate Limiting** and security headers
- **Prometheus Metrics** for monitoring
- **GraphQL** API endpoint alongside REST APIs
- **WebHooks** for event notifications

### AI Modules

#### 1. Market Analysis
- Real-time sentiment analysis using NLP (TextBlob & Transformers)
- Trend detection with statistical models
- Volatility modeling and risk assessment
- Market indicators and technical analysis

#### 2. Financial Forecasting
- ARIMA, Prophet, and regression models
- Multi-period forecasts with confidence intervals
- Scenario-based predictions
- Accuracy metrics (MAPE, RMSE, MAE)

#### 3. Competitive Intelligence
- Competitor tracking and profiling
- Market positioning analysis
- Pricing and product comparison
- SWOT analysis automation

#### 4. Customer Behavior Analytics
- Customer segmentation (K-means, hierarchical clustering)
- Churn prediction with machine learning
- Lifetime Value (LTV) calculation
- Behavioral trend analysis

#### 5. Recommendation Engine
- Strategic recommendations with explanations
- Priority-based suggestions
- Impact analysis
- Feedback loop for continuous improvement

### API Endpoints

#### Authentication (`/api/v1/auth`)
- `POST /register` - User registration
- `POST /login` - Login with JWT tokens
- `POST /refresh` - Refresh access token
- `GET /me` - Get current user info
- `POST /password-reset` - Request password reset
- `POST /change-password` - Change password
- `POST /logout` - Logout

#### Market Analysis (`/api/v1/market`)
- `POST /analyze` - Comprehensive market analysis
- `GET /sentiment/{symbol}` - Sentiment analysis
- `GET /trends` - Current market trends
- `GET /volatility/{symbol}` - Volatility metrics

#### Financial Forecasting (`/api/v1/forecasts`)
- `POST /create` - Generate new forecast
- `GET /{forecast_id}` - Get forecast by ID
- `POST /scenarios` - Create scenario variations
- `GET /` - List all forecasts

#### Competitive Intelligence (`/api/v1/competitors`)
- `POST /` - Add new competitor
- `GET /{competitor_id}/analysis` - Competitor analysis
- `GET /` - List all competitors
- `GET /matrix` - Competitive positioning matrix

#### Customer Analytics (`/api/v1/customers`)
- `GET /segments` - Customer segmentation
- `GET /{customer_id}/churn` - Churn risk prediction
- `GET /{customer_id}/ltv` - Lifetime value calculation
- `GET /` - List customers
- `GET /behavior/trends` - Behavior trends

#### Recommendations (`/api/v1/recommendations`)
- `GET /` - Get strategic recommendations
- `GET /{recommendation_id}` - Get specific recommendation
- `POST /feedback` - Submit feedback
- `POST /generate` - Generate new recommendations

#### Data Management (`/api/v1/data`)
- `POST /upload` - Upload data files (CSV, JSON, Excel)
- `GET /search` - Search across data
- `GET /uploads` - List uploaded files

#### Data Export (`/api/v1/export`)
- `GET /csv/{data_type}` - Export as CSV
- `GET /json/{data_type}` - Export as JSON
- `GET /excel/{data_type}` - Export as Excel
- `GET /pdf/{report_type}` - Export as PDF

#### Webhooks (`/api/v1/webhooks`)
- `POST /` - Create webhook
- `GET /` - List webhooks
- `GET /{webhook_id}` - Get webhook
- `PUT /{webhook_id}` - Update webhook
- `DELETE /{webhook_id}` - Delete webhook

#### GraphQL
- `/api/v1/graphql` - GraphQL endpoint with queries and mutations

### Frontend (React + TypeScript)

Premium dashboard application with:
- Responsive navigation and layouts
- Drag-and-drop customizable dashboards
- Interactive data visualizations (Plotly)
- Real-time WebSocket updates
- Authentication UI with MFA support
- Dark mode and accessibility (WCAG 2.1 AA)
- Multi-language support (i18n)
- PDF export functionality
- Component library with Storybook

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose (optional)

## 🛠️ Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd business-ai-assistant

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up --build
```

The API will be available at `http://localhost:8000`
The frontend will be available at `http://localhost:3000`

### Manual Installation

#### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -c "from ai_business_assistant.shared.database import init_db; import asyncio; asyncio.run(init_db())"

# Run the application
python -m ai_business_assistant.main
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
# Application
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/business_ai

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# External APIs (optional)
MARKET_DATA_API_KEY=your-api-key
NEWS_API_KEY=your-api-key
```

## 📚 API Documentation

Once the application is running, visit:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_business_assistant --cov-report=html

# Run specific test file
pytest tests/test_api_auth.py

# Frontend tests
cd frontend
npm test
```

## 🏗️ Architecture

```
business-ai-assistant/
├── ai_business_assistant/          # Backend application
│   ├── api/                        # API endpoints
│   │   ├── auth.py                 # Authentication
│   │   ├── market.py               # Market analysis
│   │   ├── forecasting.py          # Forecasting
│   │   ├── competitive.py          # Competitive intelligence
│   │   ├── customer.py             # Customer analytics
│   │   ├── recommendations.py      # Recommendations
│   │   ├── data.py                 # Data management
│   │   ├── export.py               # Data export
│   │   ├── webhooks.py             # Webhooks
│   │   └── graphql_app.py          # GraphQL
│   ├── models/                     # Database models
│   ├── ai_modules/                 # AI/ML modules
│   ├── shared/                     # Shared utilities
│   ├── config.py                   # Configuration
│   └── main.py                     # Main application
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── components/             # Reusable components
│   │   ├── pages/                  # Page components
│   │   ├── dashboards/             # Dashboard views
│   │   └── app/                    # App configuration
├── tests/                          # Test suite
├── docs/                           # Documentation
├── docker/                         # Docker configurations
└── infra/                          # Infrastructure as code
```

## 🚢 Deployment

### Production Checklist

1. **Update environment variables**
   - Set `ENVIRONMENT=production`
   - Use strong `SECRET_KEY`
   - Configure production database
   - Set up SSL/TLS

2. **Database**
   - Run migrations
   - Set up backups
   - Configure connection pooling

3. **Caching**
   - Configure Redis cluster
   - Set appropriate TTL values

4. **Security**
   - Enable rate limiting
   - Configure CORS properly
   - Set up security headers
   - Enable HTTPS

5. **Monitoring**
   - Set up Prometheus metrics
   - Configure logging
   - Set up alerts

### Docker Production Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📖 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Developer Setup](docs/dev-setup.md)
- [User Guide](docs/user-guide.md)
- [Data Pipelines](docs/data_pipelines.md)
- [Warehouse Schema](docs/warehouse_schema.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is proprietary and confidential.

## 🆘 Support

For support and questions:
- Email: support@business-ai-assistant.com
- Documentation: https://docs.business-ai-assistant.com
- Issue Tracker: GitHub Issues

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Scikit-learn, TensorFlow, and PyTorch for ML capabilities
- React and TypeScript for the frontend
- All open-source contributors
