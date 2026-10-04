"""Weather facts from Open-Meteo (free, no API key). Returns a plain-English fact string used as context."""
import requests


def geocode(place):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1}, timeout=10).json()
    if not r.get("results"):
        return None
    x = r["results"][0]
    return x["latitude"], x["longitude"], x["name"]


def forecast_facts(lat, lon, place="your area", days=3):
    d = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": lat, "longitude": lon, "forecast_days": days, "timezone": "auto",
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min"}).json()["daily"]
    lines = []
    for i, day in enumerate(d["time"]):
        lines.append(f"{day}: rain chance {d['precipitation_probability_max'][i]} percent, "
                     f"rainfall {d['precipitation_sum'][i]} mm, temperature {d['temperature_2m_min'][i]} to "
                     f"{d['temperature_2m_max'][i]} degrees Celsius.")
    return f"Weather forecast for {place} (source: Open-Meteo): " + " ".join(lines)
