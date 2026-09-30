import pytest
from unittest.mock import patch, MagicMock
import requests

from scripts.base_tools import search_location

LONDON_LOC = {
    "name": "London",
    "country": "United Kingdom",
    "admin1": "England",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "timezone": "Europe/London",
}

PARIS_LOC = {
    "name": "Paris",
    "country": "France",
    "admin1": "Île-de-France",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "timezone": "Europe/Paris",
}

PARIS_TX_LOC = {
    "name": "Paris",
    "country": "United States",
    "admin1": "Texas",
    "latitude": 33.6609,
    "longitude": -95.5555,
    "timezone": "America/Chicago",
}

def make_response(json_data=None, status_code=200, raise_for_status=None):
    """Build a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp

def test_empty_string_returns_error():
    # Act
    result = search_location.invoke({"city_name": ""})

    # Assert
    assert result == "City name cannot be empty."

def test_whitespace_only_returns_error():
    # Act
    result = search_location.invoke({"city_name": "   "})

    # Assert
    assert result == "City name cannot be empty."

@patch("requests.get")
def test_single_result_returns_formatted_string(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    result = search_location.invoke({"city_name": "London"})
    # Should NOT start with "Multiple locations"
    assert "Multiple locations" not in result
    assert "London" in result

@patch("requests.get")
def test_multiple_results_lists_all_with_index(mock_get):
    mock_get.return_value = make_response(
        {"results": [PARIS_LOC, PARIS_TX_LOC]}
    )
    result = search_location.invoke({"city_name": "Paris"})
    assert "Multiple locations found for 'Paris':" in result
    assert "  1." in result
    assert "  2." in result

@patch("requests.get")
def test_no_results_returns_not_found_message(mock_get):
    mock_get.return_value = make_response({"results": []})
    result = search_location.invoke({"city_name": "Xyznonexistentcity"})
    assert "No coordinates found for 'Xyznonexistentcity'." == result

@patch("requests.get")
def test_missing_results_key_treated_as_empty(mock_get):
    # API returns JSON with no "results" key
    mock_get.return_value = make_response({})
    result = search_location.invoke({"city_name": "London"})
    assert "No coordinates found" in result

@patch("requests.get")
def test_city_name_is_stripped_before_request(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    search_location.invoke({"city_name": "  London  "})
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["name"] == "London"

@patch("requests.get")
def test_count_param_forwarded(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    search_location.invoke({"city_name": "London", "count": 3})
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["count"] == 3

@patch("requests.get")
def test_default_count_is_five(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    search_location.invoke({"city_name": "London"})
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["count"] == 5

@patch("requests.get")
def test_correct_url_is_called(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    search_location.invoke({"city_name": "London"})
    args, _ = mock_get.call_args
    assert args[0] == "https://geocoding-api.open-meteo.com/v1/search"

@patch("requests.get")
def test_timeout_kwarg_is_set(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    search_location.invoke({"city_name": "London"})
    _, kwargs = mock_get.call_args
    assert kwargs.get("timeout") == 10

@patch("requests.get")
def test_timeout_returns_friendly_message(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()
    result = search_location.invoke({"city_name": "London"})
    assert "timed out" in result.lower()
    assert "London" in result

@patch("requests.get")
def test_http_error_returns_api_error_message(mock_get):
    http_err = requests.exceptions.HTTPError("404 Client Error")
    mock_get.return_value = make_response(
        raise_for_status=http_err
    )
    result = search_location.invoke({"city_name": "London"})
    assert "API error" in result
    assert "London" in result

@patch("requests.get")
def test_connection_error_returns_network_error_message(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("refused")
    result = search_location.invoke({"city_name": "London"})
    assert "Network error" in result
    assert "London" in result

@patch("requests.get")
def test_generic_request_exception_returns_network_error(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("generic")
    result = search_location.invoke({"city_name": "London"})
    assert "Network error" in result

@patch("requests.get")
def test_invalid_json_returns_parse_error(mock_get):
    mock_resp = make_response()
    mock_resp.json.side_effect = ValueError("No JSON")
    mock_get.return_value = mock_resp
    result = search_location.invoke({"city_name": "London"})
    assert "Failed to parse API response" in result

@patch("requests.get")
def test_timeout_message_includes_city_name(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()
    result = search_location.invoke({"city_name": "Tokyo"})
    assert "Tokyo" in result

@patch("requests.get")
def test_http_error_message_includes_city_name(mock_get):
    mock_get.return_value = make_response(
        raise_for_status=requests.exceptions.HTTPError("500")
    )
    result = search_location.invoke({"city_name": "Berlin"})
    assert "Berlin" in result

@patch("requests.get")
def test_exactly_one_result_no_index_prefix(mock_get):
    mock_get.return_value = make_response({"results": [LONDON_LOC]})
    result = search_location.invoke({"city_name": "London"})
    assert not result.startswith("1.")
    assert not any(line.strip().startswith("1.") for line in result.splitlines())

@patch("requests.get")
def test_multiple_results_count_matches_data(mock_get):
    locs = [PARIS_LOC, PARIS_TX_LOC, LONDON_LOC]
    mock_get.return_value = make_response({"results": locs})
    result = search_location.invoke({"city_name": "Paris", "count": 3})
    assert "  1." in result
    assert "  2." in result
    assert "  3." in result

@patch("requests.get")
def test_special_characters_in_city_name(mock_get):
    mock_get.return_value = make_response({"results": []})
    result = search_location.invoke({"city_name": "São Paulo"})
    assert "São Paulo" in result

@patch("requests.get")
def test_unicode_city_name_not_mangled(mock_get):
    mock_get.return_value = make_response({"results": []})
    search_location.invoke({"city_name": "Zürich"})
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["name"] == "Zürich"








