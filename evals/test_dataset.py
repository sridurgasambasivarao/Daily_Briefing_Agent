# To robustly test tool calling, you must evaluate three core behaviors: 
# Exact Match (straightforward mapping), 
# Chained Execution (output of tool A becomes input of tool B), and 
# Negative Constraints (preventing hallucinated tool selection).
# Explicitly split the keys inside your scenarios so your runner knows which keys to evaluate mathematically and which keys to hand off to the LLM judge.

SCENARIOS = [
    {
        "id": "scenario_1_exact_search",
        "input": "Find the latest updates on the tech conference happening in San Francisco today.",        
        "steps": [
            {
                "expected_tool": "google-calendar_list-events",
                "deterministic_args": {"calendarId": "primary"},
                "semantic_args": {
                    "timeMin": "Must accurately represent the lower bound of today in format ISO 8601",
                    "timeMax": "Must accurately represent the upper bound of today in format ISO 8601"
                }           
            },
            {
                "expected_tool": "gmail_search_emails",
                "deterministic_args": {},
                "semantic_args": {
                    "query": "Must search for terms like tech conference, San Francisco"
                }              
            },
            {
                "expected_tool": "web_search",
                "deterministic_args": {},
                "semantic_args": {
                    "query": "latest updates tech conference San Francisco today"
                }  
            }
        ]
    },
    {
        "id": "scenario_2_chained_weather",
        "input": "What's the current weather like in New York right now?",        
        "steps": [
            {
                "expected_tool": "search_location",
                "deterministic_args": {"city_name": "New York"},
                "semantic_args": {}
            },
            {
                "expected_tool": "get_weather_by_coords",
                # Coordinates are exact floats, perfect for deterministic checks
                "deterministic_args": {"latitude": 40.7128, "longitude": -74.0060},
                "semantic_args": {}
            }
        ]
    },
    {
        "id": "scenario_3_calendar_mcp",
        "input": "Schedule a briefing meeting with team@company.com tomorrow at 10 AM.",
        "steps": [
            {
                "expected_tool": "google-calendar_insert_event",
                "deterministic_args": {
                    "summary": "Briefing Meeting",
                    "attendees": ["team@company.com"]
                },
                "semantic_args": {
                    "start_time": "Must accurately represent tomorrow at 10:00 AM."
                }
            }
        ]
    },
    {
        "id": "scenario_4_negative_conversational",
        "input": "Good morning! Hope you have a great day.",
        "steps": [
            {
                "expected_tool": None, # Agent should reply conversationally without hitting any tools
                "deterministic_args": {},
                "semantic_args": {}
            }
        ]
    }
]