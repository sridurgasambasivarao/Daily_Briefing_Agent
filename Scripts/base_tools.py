import os
import requests
from ollama import Client
from langchain.tools import tool
from requests.exceptions import HTTPError, Timeout

@tool
def web_search(query: str):
    """
    Perform a live web search using Ollama Cloud Web Search API for real-time information 
    and news.

    Input:
        query: search query string

    Output:
        JSON string of top results (max_results=2).
    """    

    # 1. Explicitly initialize a dedicated remote client pointing to the cloud service
    cloud_client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    )
    
    response = cloud_client.web_search(query=query, max_results=2)
    
    return 


@tool
def search_location(city_name: str) -> str:
    """Retrieves timezone, latitude and longitude coordinates for a given city name using 
    Open-Meteo's Geocoding API."""

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "results" not in data or not data["results"]:
        return f"No coordinates found for {city_name}."
    
    loc = data["results"][0]
    name = loc.get("name")
    country = loc.get("country", "")
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    timezone = loc.get("timezone", "UTC")
    
    return f"Location: {name}, {country} | Lat: {lat}, Lon: {lon} | Timezone: {timezone}"

@tool
def get_weather_by_coords(latitude: float, longitude: float) -> str:    

    """Get current weather information for a specific latitude and longitude."""

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
