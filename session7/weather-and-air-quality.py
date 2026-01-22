import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-and-air-quality")

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def aqi_category_us(aqi: float | int | None) -> str:
    if aqi is None:
        return "Unknown"
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


async def geocode_city(city: str) -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find a location for city='{city}'")
    return results[0]


def loc_summary(loc: dict) -> dict:
    return {
        "name": loc.get("name"),
        "admin1": loc.get("admin1"),
        "country": loc.get("country"),
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
    }


@mcp.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city (Open-Meteo, no API key). Returns JSON."""
    loc = await geocode_city(city)
    lat, lon = loc["latitude"], loc["longitude"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,wind_speed_10m,weather_code",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    current = data.get("current", {}) or {}
    code = current.get("weather_code")
    condition = WMO_WEATHER_CODES.get(code, f"Unknown (code {code})")

    out = {
        "tool": "get_weather",
        "source": "open-meteo.com",
        "location": loc_summary(loc),
        "current": {
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": code,
            "condition": condition,
        },
    }
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
async def get_air_quality(city: str) -> str:
    """Get current air quality for a city (Open-Meteo Air Quality API, no key). Returns JSON."""
    loc = await geocode_city(city)
    lat, lon = loc["latitude"], loc["longitude"]

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm10,pm2_5,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    current = data.get("current", {}) or {}
    aqi = current.get("us_aqi")

    out = {
        "tool": "get_air_quality",
        "source": "open-meteo.com (air-quality)",
        "location": loc_summary(loc),
        "current": {
            "time": current.get("time"),
            "us_aqi": aqi,
            "us_aqi_category": aqi_category_us(aqi),
            "pm10_ug_m3": current.get("pm10"),
            "pm2_5_ug_m3": current.get("pm2_5"),
            "ozone_ug_m3": current.get("ozone"),
            "nitrogen_dioxide_ug_m3": current.get("nitrogen_dioxide"),
            "sulphur_dioxide_ug_m3": current.get("sulphur_dioxide"),
            "carbon_monoxide_ug_m3": current.get("carbon_monoxide"),
        },
    }
    return json.dumps(out, ensure_ascii=False)


if __name__ == "__main__":
    # MCP over stdio
    mcp.run()