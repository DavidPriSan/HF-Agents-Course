from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool,LiteLLMModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Tools

@tool
def get_number_rhyme(number:int)-> str:
    """A tool that gives a funny sentence that rhymes with the given number, in spanish.
    Args:
        number: The number the sentence will rhyme with (from 1 to 9).
    """
    rhymes = {
        1: "Uno: con mi p**e te vacuno.",
        2: "Dos: te la m**o como un dios.",
        3: "Tres: te la m**o del revés.",
        4: "Cuatro: en tu c**o mi aparato.",
        5: "Cinco: por el c**o te la hinco.",
        6: "Seis: que te la m**a el rey",
        7: "Siete: por el c**o se te mete.",
        8: "Ocho: por el c**o te la antocho.",
        9: "Nueve: el c**o se te mueve.",
    }
    return rhymes.get(number, f"I do not have a rhyme for number {number}, only from 1 to 9.")

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

final_answer = FinalAnswerTool()

# Model

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:7b",
    api_base="http://localhost:11434",
    num_ctx=8192,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Agent

agent = CodeAgent(
    model=model,
    tools=[final_answer, get_number_rhyme, get_current_time_in_timezone],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()