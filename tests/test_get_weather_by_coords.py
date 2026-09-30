from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError, Timeout, ConnectionError

from scripts.base_tools import get_weather_by_coords

def make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response with the given JSON payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()  # no-op by default (success)
    return mock_resp


VALID_API_RESPONSE = {
    "current": {
        "temperature_2m": 15.3,
        "wind_speed_10m": 12.1,
        "weather_code": 1,
    },
    "current_units": {
        "temperature_2m": "°C",
        "wind_speed_10m": "km/h",
    },
}

def test_latitude_too_low():
    result = get_weather_by_coords.invoke({"latitude": -91.0, "longitude": 0.0})
    assert "Invalid latitude" in result

def test_latitude_too_high():
    result = get_weather_by_coords.invoke({"latitude": 91.0, "longitude": 0.0})
    assert "Invalid latitude" in result

def test_latitude_boundary_min():
    """Exact boundary -90.0 is valid and should not return a validation error."""
    with patch("requests.get", return_value=make_response(VALID_API_RESPONSE)):
        result = get_weather_by_coords.invoke({"latitude": -90.0, "longitude": 0.0})
    assert "Invalid latitude" not in result

def test_latitude_boundary_max():
    with patch("requests.get", return_value=make_response(VALID_API_RESPONSE)):
        result = get_weather_by_coords.invoke({"latitude": 90.0, "longitude": 0.0})
    assert "Invalid latitude" not in result

def test_longitude_too_low():
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": -181.0})
    assert "Invalid longitude" in result

def test_longitude_too_high():
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 181.0})
    assert "Invalid longitude" in result

def test_longitude_boundary_min():
    with patch("requests.get", return_value=make_response(VALID_API_RESPONSE)):
        result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": -180.0})
    assert "Invalid longitude" not in result

def test_longitude_boundary_max():
    with patch("requests.get", return_value=make_response(VALID_API_RESPONSE)):
        result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 180.0})
    assert "Invalid longitude" not in result




@patch("requests.get")
def test_returns_temperature(mock_get):
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    result = get_weather_by_coords.invoke({"latitude": 51.5, "longitude": -0.1})
    assert "15.3" in result
    assert "°C" in result

@patch("requests.get")
def test_returns_wind_speed(mock_get):
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    result = get_weather_by_coords.invoke({"latitude": 51.5, "longitude": -0.1})
    assert "12.1" in result
    assert "km/h" in result

@patch("requests.get")
def test_returns_weather_condition(mock_get):
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    result = get_weather_by_coords.invoke({"latitude": 51.5, "longitude": -0.1})
    # weather_code 1 → "Mainly clear"
    assert "Mainly clear" in result

@patch("requests.get")
def test_unknown_weather_code_falls_back(mock_get):
    data = {**VALID_API_RESPONSE, "current": {**VALID_API_RESPONSE["current"], "weather_code": 999}}
    mock_get.return_value = make_response(data)
    result = get_weather_by_coords.invoke({"latitude": 51.5, "longitude": -0.1})
    assert "Unknown condition" in result

@patch("requests.get")
def test_uses_api_units(mock_get):
    """Units must come from current_units, not be hardcoded."""
    data = {
        "current": {"temperature_2m": 59.0, "wind_speed_10m": 7.5, "weather_code": 0},
        "current_units": {"temperature_2m": "°F", "wind_speed_10m": "mph"},
    }
    mock_get.return_value = make_response(data)
    result = get_weather_by_coords.invoke({"latitude": 40.7, "longitude": -74.0})
    assert "°F" in result
    assert "mph" in result

@patch("requests.get")
def test_correct_api_url_called(mock_get):
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    get_weather_by_coords.invoke({"latitude": 48.85, "longitude": 2.35})
    call_args = mock_get.call_args
    assert "open-meteo.com" in call_args[0][0]

@patch("requests.get")
def test_coords_passed_as_params(mock_get):
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    get_weather_by_coords.invoke({"latitude": 48.85, "longitude": 2.35})
    params = mock_get.call_args[1]["params"]
    assert params["latitude"] == 48.85
    assert params["longitude"] == 2.35

@patch("requests.get")
def test_missing_temperature_returns_unavailable(mock_get):
    data = {
        "current": {"wind_speed_10m": 5.0, "weather_code": 0},
        "current_units": {},
    }
    mock_get.return_value = make_response(data)
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "unavailable" in result.lower()

@patch("requests.get")
def test_missing_wind_speed_returns_unavailable(mock_get):
    data = {
        "current": {"temperature_2m": 20.0, "weather_code": 0},
        "current_units": {},
    }
    mock_get.return_value = make_response(data)
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "unavailable" in result.lower()

@patch("requests.get")
def test_empty_current_block_returns_unavailable(mock_get):
    mock_get.return_value = make_response({"current": {}, "current_units": {}})
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "unavailable" in result.lower()


@patch("requests.get")
def test_http_error_404(mock_get):
    mock_resp = make_response({}, status_code=404)
    mock_resp.raise_for_status.side_effect = HTTPError("404 Not Found")
    mock_get.return_value = mock_resp
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "HTTP error" in result
    assert "404" in result

@patch("requests.get")
def test_http_error_500(mock_get):
    mock_resp = make_response({}, status_code=500)
    mock_resp.raise_for_status.side_effect = HTTPError("500 Server Error")
    mock_get.return_value = mock_resp
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "HTTP error" in result

@patch("requests.get", side_effect=Timeout())
def test_timeout(mock_get):
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "timed out" in result.lower()

@patch("requests.get", side_effect=ConnectionError())
def test_connection_error(mock_get):
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "connect" in result.lower()

@patch("requests.get", side_effect=ValueError("bad json"))
def test_unexpected_error_does_not_expose_internals(mock_get):
    result = get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert "unexpected error" in result.lower()
    assert "ValueError" not in result  # internals not leaked to user

@patch("requests.get")
def test_timeout_kwarg_is_set(mock_get):
    """Ensure the call always sets a timeout so it can't hang forever."""
    mock_get.return_value = make_response(VALID_API_RESPONSE)
    get_weather_by_coords.invoke({"latitude": 0.0, "longitude": 0.0})
    assert mock_get.call_args[1].get("timeout") is not None