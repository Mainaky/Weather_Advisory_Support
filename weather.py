"""
weather.py - Open-Meteo Geocoding and Weather Data Fetcher

Fetches live weather data from Open-Meteo's free, public APIs:
- Geocoding: https://geocoding-api.open-meteo.com/v1/search?name=<city>
- Forecast:  https://api.open-meteo.com/v1/forecast

Adheres strictly to the assignment specifications:
- Requests explicit current, hourly, and daily fields.
- Uses wind_speed_unit=kmh and timezone=auto.
- Returns verified flat dictionary of weather facts or raises clean, explicit exceptions.
"""

import requests
from typing import Dict, Any, Optional
from datetime import datetime


class WeatherServiceError(Exception):
    """Base exception for weather service operations."""
    pass


class GeocodingError(WeatherServiceError):
    """Raised when geocoding fails or a city cannot be resolved."""
    pass


class WeatherAPIError(WeatherServiceError):
    """Raised when the forecast API request fails or returns an error."""
    pass


# Explicit fields required by assignment specification
CURRENT_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
    "pressure_msl",
    "relative_humidity_2m",
]

HOURLY_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
    "pressure_msl",
    "relative_humidity_2m",
]

DAILY_FIELDS = [
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_gusts_10m_max",
    "uv_index_max",
]


def geocode_city(city_name: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Resolves a human city name to latitude, longitude, and display metadata.
    Returns the first matching result.
    Raises GeocodingError if not found or if the network request fails.
    """
    cleaned_name = city_name.strip()
    if not cleaned_name:
        raise GeocodingError("City name cannot be empty.")

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": cleaned_name, "count": 1, "language": "en", "format": "json"}

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise GeocodingError(f"Network error during geocoding for '{cleaned_name}': {exc}") from exc
    except ValueError as exc:
        raise GeocodingError(f"Invalid JSON response from geocoding service: {exc}") from exc

    results = data.get("results")
    if not results or len(results) == 0:
        raise GeocodingError(f"Location not found: '{cleaned_name}'. Please verify spelling or provide a nearby city.")

    first = results[0]
    return {
        "name": first.get("name", cleaned_name),
        "latitude": float(first["latitude"]),
        "longitude": float(first["longitude"]),
        "country": first.get("country", ""),
        "admin1": first.get("admin1", ""),
    }


def fetch_raw_forecast(latitude: float, longitude: float, timeout: int = 10) -> Dict[str, Any]:
    """
    Queries the Open-Meteo forecast endpoint for current, hourly, and daily metrics.
    Raises WeatherAPIError on failure.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
        "current": ",".join(CURRENT_FIELDS),
        "hourly": ",".join(HOURLY_FIELDS),
        "daily": ",".join(DAILY_FIELDS),
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Failed to fetch forecast from Open-Meteo: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError(f"Invalid JSON response from Open-Meteo forecast API: {exc}") from exc


def extract_weather_facts(
    raw_data: Dict[str, Any],
    time_ref: Optional[str] = None,
    location_name: str = "",
) -> Dict[str, Any]:
    """
    Flattens raw Open-Meteo forecast data into a simple dictionary of facts for the SOP evaluator.
    Handles temporal references (e.g. 'evening', 'tomorrow', 'now', 'today').
    """
    current = raw_data.get("current", {})
    daily = raw_data.get("daily", {})
    hourly = raw_data.get("hourly", {})

    # Default daily metrics (first day / today)
    daily_precip_sum = float(daily.get("precipitation_sum", [0.0])[0] if daily.get("precipitation_sum") else 0.0)
    precip_prob_max = float(daily.get("precipitation_probability_max", [0.0])[0] if daily.get("precipitation_probability_max") else 0.0)
    wind_gusts_max = float(daily.get("wind_gusts_10m_max", [0.0])[0] if daily.get("wind_gusts_10m_max") else 0.0)
    uv_max = float(daily.get("uv_index_max", [0.0])[0] if daily.get("uv_index_max") else 0.0)

    # Base baseline facts from current observation
    facts: Dict[str, Any] = {
        "temperature_2m": float(current.get("temperature_2m", 0.0)),
        "apparent_temperature": float(current.get("apparent_temperature", 0.0)),
        "wind_speed_10m": float(current.get("wind_speed_10m", 0.0)),
        "wind_gusts_10m": float(current.get("wind_gusts_10m", 0.0)),
        "precipitation": float(current.get("precipitation", 0.0)),
        "precipitation_probability": float(current.get("precipitation_probability", 0.0)),
        "uv_index": float(current.get("uv_index", 0.0)),
        "weather_code": int(current.get("weather_code", 0)),
        "pressure_msl": float(current.get("pressure_msl", 1013.25)),
        "relative_humidity_2m": float(current.get("relative_humidity_2m", 50.0)),
        "daily_precipitation_sum": daily_precip_sum,
        "precipitation_probability_max": precip_prob_max,
        "wind_gusts_10m_max": wind_gusts_max,
        "uv_index_max": uv_max,
        "time": current.get("time", ""),
        "hour": 12,  # default fallback hour
        "location_name": location_name,
    }

    # Extract hour from current time if available (e.g., '2026-09-04T14:30')
    if facts["time"] and "T" in facts["time"]:
        try:
            facts["hour"] = int(facts["time"].split("T")[1].split(":")[0])
        except (ValueError, IndexError):
            facts["hour"] = 12

    # If user specifically asked for 'evening' or 'tomorrow', select from hourly series
    normalized_ref = (time_ref or "").lower().strip()
    hourly_times = hourly.get("time", [])

    if normalized_ref == "evening" and hourly_times:
        # Target 18:00 (6 PM) of today
        target_indices = [i for i, t in enumerate(hourly_times[:24]) if t.endswith("T18:00") or t.endswith("T19:00")]
        idx = target_indices[0] if target_indices else (18 if len(hourly_times) > 18 else 0)

        facts["hour"] = 18
        facts["time"] = hourly_times[idx]
        for field in HOURLY_FIELDS:
            if field in hourly and len(hourly[field]) > idx:
                val = hourly[field][idx]
                if field == "weather_code":
                    facts[field] = int(val) if val is not None else 0
                else:
                    facts[field] = float(val) if val is not None else 0.0

    elif normalized_ref == "tomorrow" and len(hourly_times) >= 48:
        # Tomorrow midday 12:00 (index ~36)
        target_indices = [i for i, t in enumerate(hourly_times[24:48], start=24) if t.endswith("T12:00")]
        idx = target_indices[0] if target_indices else 36

        facts["hour"] = 12
        facts["time"] = hourly_times[idx]
        for field in HOURLY_FIELDS:
            if field in hourly and len(hourly[field]) > idx:
                val = hourly[field][idx]
                if field == "weather_code":
                    facts[field] = int(val) if val is not None else 0
                else:
                    facts[field] = float(val) if val is not None else 0.0

        # Also update daily metrics for tomorrow (day 1)
        if len(daily.get("precipitation_sum", [])) > 1:
            facts["daily_precipitation_sum"] = float(daily["precipitation_sum"][1])
        if len(daily.get("precipitation_probability_max", [])) > 1:
            facts["precipitation_probability_max"] = float(daily["precipitation_probability_max"][1])
        if len(daily.get("wind_gusts_10m_max", [])) > 1:
            facts["wind_gusts_10m_max"] = float(daily["wind_gusts_10m_max"][1])
        if len(daily.get("uv_index_max", [])) > 1:
            facts["uv_index_max"] = float(daily["uv_index_max"][1])

    # Round floats neatly to 1 decimal place for stable verification
    for k, v in list(facts.items()):
        if isinstance(v, float):
            facts[k] = round(v, 1)

    return facts


def get_live_weather_facts(location_name: str, time_ref: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience orchestrator: Geocodes the location, fetches forecast, and returns parsed facts.
    """
    geo = geocode_city(location_name)
    raw = fetch_raw_forecast(geo["latitude"], geo["longitude"])
    display_name = f"{geo['name']}, {geo['country']}".strip(", ")
    return extract_weather_facts(raw, time_ref=time_ref, location_name=display_name)
