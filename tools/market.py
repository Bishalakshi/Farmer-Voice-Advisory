"""Market price facts. Uses data/market_prices.csv (replace with Agmarknet / data.gov.in feed if you have an API key)."""
import pandas as pd
import config


def price_facts(crop):
    df = pd.read_csv(config.MARKET_FILE)
    g = df[df.crop.str.lower() == str(crop).lower()].sort_values("date")
    if g.empty:
        return None
    r = g.iloc[-1]
    return (f"Latest price for {r.crop} at {r.market} on {r.date}: modal {int(r.modal_price)}, "
            f"minimum {int(r.min_price)}, maximum {int(r.max_price)} {r.unit} (source: market file).")
