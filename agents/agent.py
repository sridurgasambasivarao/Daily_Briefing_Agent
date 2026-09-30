"""Daily Briefing Agent with MCP Tools."""
"""Use gmail, google calendar, web search and weather for daily morning update"""

import sys
import os
import logging

from typing import List, Optional
from langchain_core.tools import BaseTool

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from scripts import base_tools, prompts, utils

from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

logger = logging.getLogger(__name__)

async def get_tools(mcp_client: Optional[MultiServerMCPClient] = None) -> List[BaseTool]:     
    """
    Fetches and filters tools. 
    Accepts an optional pre-configured mcp_client (Dependency Injection).
    """
    tools = []

    # If no client is injected, create the production one (default behavior)
    if mcp_client is None:
        mcp_config = utils.load_mcp_config("gmail", "google-calendar")
        mcp_client = MultiServerMCPClient(mcp_config) 

    # Fetch gmail tools safely
    try:
        logger.info(f"Connecting and loading gmail tools from MCP server")
        gmail_tools = await mcp_client.get_tools(server_name="gmail")
        tools.extend(gmail_tools)
    except Exception as e:
        logger.error(f"gmail server down '{e}'. Skipping.")        

    # Fetch google calendar tools safely
    try:
        logger.info(f"Connecting and loading google calendar tools from MCP server")
        calendar_tools = await mcp_client.get_tools(server_name="google-calendar")
        tools.extend(calendar_tools)
    except Exception as e:         
        logger.error(f"google calendar server down '{e}'. Skipping.")      

    if tools:
        # Filter tools that work with Gemini
        filter_tools = ['delete_email', 'batch_modify_emails', 'batch_delete_emails', 'delete_label', 'delete_filter']
            
        safe_tools = [tool for tool in tools if tool.name not in filter_tools]
        logger.info(f"Loaded {len(safe_tools)} Tools")
        logger.info(f"Tools Available\n{[tool.name for tool in safe_tools]}")
        
        return safe_tools + [base_tools.web_search, base_tools.search_location, 
                             base_tools.get_weather_by_coords]
    else:
        return tools
    

async def get_agent(explicit_tools: Optional[List[BaseTool]] = None, mcp_client: Optional[MultiServerMCPClient] = None):
    """
    Builds the agent.
    Allows injecting either a final list of tools (explicit_tools) or a custom client wrapper.
    """
    #  If explicit mock tools are passed directly (best for unit tests), use them.
    #  If an injected client is passed, use it to resolve tools.
    #  Otherwise, fall back to default production discovery.
    if explicit_tools is not None:
        tools = explicit_tools
    else:
        tools = await get_tools(mcp_client=mcp_client)

    system_prompt = prompts.get_daily_briefing_prompt()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name=None,
                run_limit=1,  # Max 1 per conversation turn
                thread_limit=2,  # Max 2 total across all conversation turns
                exit_behavior="end" # When the limit is breached, the middleware stops executing 
                # any further tools and automatically appends an explicit ToolMessage explaining the stoppage followed by an AIMessage. Your Chat UI reads these messages seamlessly through its default streaming protocol without throwing an error
            )
        ],
    )
    
    return agent

# --- Production Usage ---
# Build the agent ONCE at startup using standard live initialization
if __name__ == "__main__":
    agent = asyncio.run(get_agent())