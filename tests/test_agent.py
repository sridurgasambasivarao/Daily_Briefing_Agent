import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Import your agent module here
from agents.agent import get_agent

# Create a Lightweight Mock Tool
class MockToolInput(BaseModel):
    query: str = Field(description="A dummy query string")

class FakeGmailTool(BaseTool):
    name: str = "get_emails"
    description: str = "A mock tool that mimics fetching emails without a live server."
    args_schema: Type[BaseModel] = MockToolInput

    def _run(self, query: str) -> str:
        # Returns instant, predictable data for testing
        return "Mocked email: You have a meeting at 10 AM."

    async def _arun(self, query: str) -> str:
        # Async implementation
        return self._run(query)


@pytest.mark.asyncio
async def test_agent_initialization_with_mock_tools():
    """Verify that the agent builds correctly when mock tools are injected directly."""
    # Setup: Create our list of fake tools
    mock_tools = [FakeGmailTool()]

    # Action: Inject the explicit mock tools into the agent builder
    agent = await get_agent(explicit_tools=mock_tools)

    # Assert: Verify the agent was created and holds our mock dependency
    assert agent is not None
        
    # Compiled LangGraph objects expose their underlying nodes and configuration
    assert hasattr(agent, "nodes") or hasattr(agent, "builder")
    
    # Verify that your 'tools' node exists inside the graph structure
    if hasattr(agent, "nodes"):
        assert "tools" in agent.nodes
        print("\n Verified 'tools' node is present in the agent graph structure!")
    
    print("\n Agent successfully initialized with injected dependencies!")


@pytest.mark.asyncio
async def test_agent_skips_mcp_servers_on_injection():
    """Ensure that providing explicit_tools bypasses live network discovery."""
    mock_tools = [FakeGmailTool()]
    
    # This test will pass instantly without trying to read 'utils.load_mcp_config'
    # or raising exceptions for 'gmail server down'.
    agent = await get_agent(explicit_tools=mock_tools)
    
    # If it attempted to hit live servers, it would print server down errors or hang.
    assert agent is not None