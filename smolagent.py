from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, tool
import numpy as np
import time
import datetime

# Local Ollama model

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:7b",
    api_base="http://localhost:11434",
    num_ctx=8192,
)

## Thematic web search

# Playlist

agent = CodeAgent(
    tools=[VisitWebpageTool(max_output_length=12000)],
    model=model,
    code_block_tags="markdown",
    max_steps=4,
)

playlist = agent.run(
    """
    Create a playlist using this chart only:
    https://chart.simpmusic.org/en/charts/es

    Instructions:
    1. Call visit_webpage exactly once with that URL.
    2. Read the chart in its displayed order.
    3. Each song title appears after "###"; the following line contains
       its artist or artists.
    4. Extract only the first 10 songs.
    5. Create a Python list named playlist containing exactly 10 unique
       strings formatted as "Song - Artist".
    6. Do not use search results, image descriptions, chart movements,
       invented songs, or any other website.
    7. Verify that len(playlist) == 10.
    8. Your final executed statement must be:
       final_answer(playlist)
    """
)

if (
    not isinstance(playlist, list)
    or len(playlist) != 10
    or not all(isinstance(song, str) and " - " in song for song in playlist)
):
    raise RuntimeError(f"Invalid playlist returned by agent: {playlist!r}")

# Costumes (based on the playlist)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    code_block_tags="markdown",
    max_steps=4,
)

agent.run(
    f"""
    This is the playlist for the party at the Wayne's mansion:
    {playlist}

    Search for costume ideas for the guests inspired by the songs in the playlist.
    Store the result in a variable named costume_ideas.
    The last executed statement MUST be final_answer(costume_ideas).
    Never return the result as a bare variable.
    """
)

## Menu with tools

# Tool to suggest a menu based on the occasion

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

agent = ToolCallingAgent(
    tools=[suggest_menu],
    model=model,
    max_steps=3,
)

agent.run(
    """
    Prepare a formal menu for the party using the available tool and store its
    result in a variable named menu.
    The last executed statement MUST be final_answer(menu).
    Never return the result as a bare variable.
    """
)

# Tool to suggest drinks based on who is comming

@tool
def suggest_drinks(guest1: str, guest2: str | None = None, guest3: str | None = None) -> str:
    """
    Suggests drinks based on the guests coming to the party (one to three guests).
    Args:
        guest1: Name of the first guest.
        guest2: Name of the second guest (optional).
        guest3: Name of the third guest (optional).
    """
    favorite_drinks = {
        "david": "Ron Naranja",
        "ruben": "Ron Cola",
        "adri": "Whisky Cola",
        "sonia": "Agua del grifo",
        "liz": "Gintonic",
        "miguel": "Ron Cola",
    }
    guests = [g for g in (guest1, guest2, guest3) if g]
    suggestions = []
    for guest in guests:
        drink = favorite_drinks.get(guest.strip().lower(), "Champagne")
        suggestions.append(f"{guest}: {drink}")
    return "; ".join(suggestions)

agent = ToolCallingAgent(
    tools=[suggest_drinks],
    model=model,
    max_steps=3,
)

agent.run(
    """
    David, Ruben and Miguel are coming to the party. Prepare the drinks for them
    using the available tool and store its result in a variable named drinks.
    The last executed statement MUST be final_answer(drinks).
    Never return the result as a bare variable.
    """
)

## Time needed

agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["datetime"],
    code_block_tags="markdown",
    max_steps=3,
)

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 60 minutes (Ruben is a bit of a boozer)
    2. Prepare the costumes for the guests - 45 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 30 minutes

    If we start right now, at what time will the party be ready?

    Calculate the total duration, add it to the current time, store the result
    in a variable named party_ready_at, and finish by calling:

    final_answer(f"The party will be ready at {party_ready_at:%H:%M}")

    Do not return a bare variable. You MUST call final_answer().
    """
)
