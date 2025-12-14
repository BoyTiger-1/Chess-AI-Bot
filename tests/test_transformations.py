import pandas as pd

from ai_business_assistant.transformations.market import normalize_market_prices


def test_normalize_market_prices_minimal():
    df = pd.DataFrame(
        [
            {
                "ticker": "aapl",
                "timestamp": "2025-01-01T00:00:00Z",
                "close": "101.5",
                "source": "unit",
            }
        ]
    )

    out = normalize_market_prices(df)

    assert list(out.columns)  # not empty
    assert out.loc[0, "symbol"] == "AAPL"
    assert out.loc[0, "close"] == 101.5
    assert str(out.loc[0, "ts"].tz) in ("UTC", "UTC+00:00")
