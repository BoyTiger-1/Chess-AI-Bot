import pandas as pd
import pytest

from ai_business_assistant.validation.schemas import validate_market_prices


def test_validate_market_prices_missing_column():
    df = pd.DataFrame([{"symbol": "AAPL"}])
    with pytest.raises(ValueError):
        validate_market_prices(df)


def test_validate_market_prices_happy_path():
    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "ts": "2025-01-01T00:00:00Z",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "source": "unit",
            }
        ]
    )
    out = validate_market_prices(df)
    assert out.shape[0] == 1
