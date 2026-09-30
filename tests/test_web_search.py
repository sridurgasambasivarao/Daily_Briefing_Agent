import json
from pydantic import BaseModel
from unittest.mock import MagicMock, patch

# Replace 'your_module' with the actual name of the file containing your function
from scripts.base_tools import web_search

# Test case for an empty or whitespace-only query
def test_web_search_empty_query():
    # Act
    result_json = web_search.invoke({"query": "   "})
    result = json.loads(result_json)
    
    # Assert
    assert "error" in result
    assert result["error"] == "Empty search query."


# Test case for when the OLLAMA_API_KEY environment variable is missing
# monkeypatch: This is a standard pytest fixture used to temporarily adjust your system environment. 
# monkeypatch.setenv fakes the existence of your API key, 
# while monkeypatch.delenv guarantees a clean state where the key is absent.
def test_web_search_missing_api_key(monkeypatch):
    # Arrange: Safely remove the environment variable for this test
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    
    # Act
    result_json = web_search.invoke({"query": "python news"})
    result = json.loads(result_json)
    
    # Assert
    assert "error" in result
    assert result["error"] == "OLLAMA_API_KEY is not set."

def test_web_search_tool_schema_is_valid():
    # Assert basic tool metadata
    assert web_search.name == "web_search"
    
    # Assert docstring parsing captured the description
    assert "Perform a live web search using Ollama Cloud Web Search API" in web_search.description
    assert "real-time information and news" in web_search.description

    # Assert the arguments schema (args_schema) is generated correctly
    assert web_search.args_schema is not None
    assert issubclass(web_search.args_schema, BaseModel)

    # Deeply inspect the JSON schema generated from the Pydantic definition
    # framework packages expose this via .args or via extracting schema from args_schema
    schema_dict = web_search.args_schema.model_json_schema()  # Use .schema() for Pydantic v1
    
    # Verify the 'query' parameter constraints
    assert "query" in schema_dict["properties"]
    assert schema_dict["properties"]["query"]["type"] == "string"
    
    # If parse_docstring successfully captured the argument description:
    assert "search query string" in schema_dict["properties"]["query"]["description"]
    
    # Verify 'query' is marked as a required input field
    assert "query" in schema_dict["required"]


# Test case for a successful API call with non-ASCII text
@patch("scripts.base_tools.Client")  # Patch 'Client' where it is imported/used
def test_web_search_success(mock_client_class, monkeypatch):
    # Arrange: Set up fake environment variable
    monkeypatch.setenv("OLLAMA_API_KEY", "fake_key")
    
    # Mock the response structure: response.results -> list of result objects
    mock_result_1 = MagicMock(title="Python 3.12 News", content="Features include better error messages.")
    mock_result_2 = MagicMock(title="Café Trends", content="Enchanteé coffee updates.")
    
    mock_response = MagicMock()
    mock_response.results = [mock_result_1, mock_result_2]
    
    # Configure the mock client instance to return our mock response
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.web_search.return_value = mock_response
    
    # Act
    result_json = web_search.invoke({"query": "café updates"})
    result = json.loads(result_json)
    
    # Assertions
    assert result["query"] == "café updates"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Python 3.12 News"
    assert result["results"][1]["content"] == "Enchanteé coffee updates."
    
    # Verify the client initialization and method call occurred as expected
    mock_client_class.assert_called_once_with(
        host="https://ollama.com",
        headers={"Authorization": "Bearer fake_key"}
    )
    mock_client_instance.web_search.assert_called_once_with(query="café updates", max_results=2)


# Test case for handling API client exceptions smoothly
@patch("scripts.base_tools.Client")
def test_web_search_api_exception(mock_client_class, monkeypatch):
    # Arrange
    monkeypatch.setenv("OLLAMA_API_KEY", "fake_key")
    
    # Force the mock client instance to raise an exception when called
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.web_search.side_effect = Exception("Connection timed out")
    
    # Act
    result_json = web_search.invoke({"query": "test query"})
    result = json.loads(result_json)
    
    # Assert
    assert "error" in result
    assert "Web search failed: Connection timed out" in result["error"]

@patch("scripts.base_tools.Client")  # Intercept the Client constructor call
def test_web_search_sends_correct_auth_header(mock_client_class, monkeypatch):
    # Arrange: Set up a distinct fake API key
    expected_api_key = "secret_ollama_token_abc123"
    monkeypatch.setenv("OLLAMA_API_KEY", expected_api_key)
    
    # Mock the return value chains so the function doesn't crash during execution
    mock_response = MagicMock()
    mock_response.results = []
    mock_client_class.return_value.web_search.return_value = mock_response
    
    # Act: Invoke the tool
    web_search.invoke({"query": "secure search"})
    
    # Assert: Verify the Client constructor was called with the correct Bearer token structure
    mock_client_class.assert_called_once_with(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {expected_api_key}"}
    )
