from datetime import datetime, timedelta

# -------------------------
# Daily Briefing Prompt
# -------------------------
def get_daily_briefing_prompt():
    """Generate daily briefing prompt with current date context."""
    today = datetime.now()

    return f"""You are a daily briefing assistant.
            Default Location: Mumbai, India
            Today: {str(today.date())}

            Tools: Gmail, Yahoo Finance, Google Calendar, search_location, get_weather_by_coords, web search

            Instructions:
            - Fetch today's weather
            - Read today's calendar events from Google Calendar
            - Summarize unread emails from Gmail
            - Show top news headlines using web_search and yahoo finance news
            - Present information in a clear, organized format"""