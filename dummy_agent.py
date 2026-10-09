from openai import OpenAI

# Local model in Ollama

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Simple chat

""" output = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "The capital of Spain is"}],
)
print(output.choices[0].message.content) """

## Weather Agent

# System prompt

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

# First message

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in Salamanca?"},
]

output = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    max_tokens=150,
    stop=["Observation:"], # Stop before calling the function
)

print(output.choices[0].message.content)

# Get weather function

def get_weather(location):
    return f"the weather in {location} is cloudy with low temperatures. \n"

# Second message

messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in Salamanca?"},
    {"role": "assistant", "content": output.choices[0].message.content + "Observation:\n" + get_weather('Salamanca')},
]

output = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    stream=False,
    max_tokens=300,
)
print(output.choices[0].message.content)