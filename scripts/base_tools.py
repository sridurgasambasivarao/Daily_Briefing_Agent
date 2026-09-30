import os
import json
import requests
import logging
from ollama import Client
from langchain.tools import tool
from requests.exceptions import HTTPError, Timeout

logger = logging.getLogger(__name__)

def _format_location(loc: dict) -> str:
    name     = loc.get("name", "Unknown")
    country  = loc.get("country", "Unknown")
    region   = loc.get("admin1", "")
    lat      = loc.get("latitude")
    lon      = loc.get("longitude")
    timezone = loc.get("timezone", "UTC")

    full_name = f"{name}, {region}, {country}" if region else f"{name}, {country}"
    fmt_coord = lambda v, label: f"{label}: {v}" if v is not None else f"{label}: N/A"
    coords    = f"{fmt_coord(lat, 'Lat')}, {fmt_coord(lon, 'Lon')}"

    return f"{full_name} | {coords} | Timezone: {timezone}"

@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """
    Perform a live web search using Ollama Cloud Web Search API for real-time information and news.

    Args:
        query: search query string

    Returns:
        JSON string of top results (max_results=2).
    """  
    query = query.strip()
    if not query:
        return json.dumps({"error": "Empty search query."})

    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return json.dumps({"error": "OLLAMA_API_KEY is not set."})  

    try:
        # Explicitly initialize a dedicated remote client pointing to the cloud service
        cloud_client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response = cloud_client.web_search(query=query, max_results=2)
    except Exception as ex:
        logger.exception("web_search failed for query=%r", query)
        return json.dumps({"error": f"Web search failed: {ex}"})
    
    results = [
        {"title": r.title, "content": r.content}
        for r in response.results
    ]

    # By default, json.dumps escapes every non-ASCII character into a \uXXXX sequence. ensure_ascii=False tells it to leave those characters as they are.
    # For non-English search results (or even English ones with accents, curly quotes, or em dashes), the escaped version can inflate token usage noticeably.
    # LLMs handle actual characters better than escape sequences.
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


@tool(parse_docstring=True)
def search_location(city_name: str, count: int = 5) -> str:
    """
    Retrieves timezone, latitude and longitude coordinates for a given city name using 
    Open-Meteo's Geocoding API.
    Returns multiple matches if the name is ambiguous.
    
    Args:
        city_name: name of city as a string
        count: max number of results to return (default 5)
    
    Returns:
        a string listing all matching locations with country, region,
        latitude, longitude and timezone.
    
    """

    if not city_name or not city_name.strip():
        return "City name cannot be empty."

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name.strip(), "count": count}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.exception("search_location did timeoutfor city_name=%r", city_name)
        return f"Request timed out while searching for '{city_name}'."
    except requests.exceptions.HTTPError as e:
        logger.exception("search_location encountered HTTP error for city_name=%r", city_name)
        return f"API error for '{city_name}': {e}"
    except requests.exceptions.RequestException as e:
        logger.exception("search_location encountered network error for city_name=%r", city_name)
        return f"Network error while searching for '{city_name}': {e}"
    except ValueError:
        logger.exception("search_location failed to parse API response for city_name=%r", city_name)
        return "Failed to parse API response."

    results = data.get("results", [])
    if not results:
        return f"No coordinates found for '{city_name}'."

    if len(results) == 1:
        loc = results[0]
        return _format_location(loc)

    # Multiple matches — list them all with an index
    lines = [f"Multiple locations found for '{city_name}':"]
    for i, loc in enumerate(results, start=1):
        lines.append(f"  {i}. {_format_location(loc)}")
    return "\n".join(lines)    

@tool(parse_docstring=True)
def get_weather_by_coords(latitude: float, longitude: float) -> str:    

    """
    Get current weather information for a specific latitude and longitude.
    
    Args:
        latitude: The latitude coordinate of the location. Must be between -90.0
          and 90.0.
        longitude: The longitude coordinate of the location. Must be between
          -180.0 and 180.0.
    
    Returns:
        a string describing current temperature and wind speed
    
    """

    url = f"https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude, "current": "temperature_2m,weather_code,wind_speed_10m"}    

    try:
        response = requests.get(url, params=params, timeout=10)
        
        # Crash immediately if the server responds with a 4xx or 5xx error
        response.raise_for_status()

        data = response.json()
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        wind = current.get("wind_speed_10m")

        return f"Current temperature: {temp}°C, Wind speed: {wind} km/h."
    
    except HTTPError as http_err:
        # Handles bad credentials, invalid cities (404), or server outages (500)
        return f"HTTP error occurred: {http_err}"
    
    except Timeout:
        # Handles situations where the weather server took longer than 10 seconds to respond
        return "The request timed out. Please try again later."
    
    except Exception as err:
        # Handles any other unexpected errors
        return "Error fetching weather data."
