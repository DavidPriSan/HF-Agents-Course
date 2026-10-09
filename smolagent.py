from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, tool
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
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(max_output_length=6000)],
    model=model,
    code_block_tags="markdown",
    max_steps=6,
)

playlist = agent.run(
    """
    Create a playlist for a party at the Wayne's mansion with the top hits in Spain right now.
    1. Use web_search to find a page with the current Spanish music chart.
    2. Use visit_webpage on one of those URLs to read the actual song titles and artists.
    3. Store exactly 10 real songs in a variable named playlist, using the
       format "Song - Artist".
    4. The last executed statement MUST be final_answer(playlist).

    Always write code. Never return the playlist as a bare variable and never
    finish without calling final_answer() inside a code block.
    """
)

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

agent = CodeAgent(
    tools=[suggest_menu],
    model=model,
    code_block_tags="markdown",
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

agent = CodeAgent(
    tools=[suggest_drinks],
    model=model,
    code_block_tags="markdown",
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
