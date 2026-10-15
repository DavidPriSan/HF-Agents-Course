import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)


# Model selection: AGENT_MODEL="ollama" (default) uses local Ollama,
# AGENT_MODEL="hf:<model_id>" uses HF Inference Providers (needs HF_TOKEN).
def build_model():
    choice = os.getenv("AGENT_MODEL", "ollama")
    if choice.startswith("hf:"):
        return InferenceClientModel(model_id=choice[3:], token=os.getenv("HF_TOKEN"))
    return LiteLLMModel(
        model_id=os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:7b"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        num_ctx=8192,
    )


@tool
def read_file(file_path: str) -> str:
    """
    Reads a local file attached to the task and returns its content as text.
    Supports spreadsheets (.xlsx, .xls, .csv) and text files (.py, .txt, .json, .md, ...).

    Args:
        file_path: Path to the file to read.
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return "\n\n".join(f"Sheet '{name}':\n{df.to_csv(index=False)}" for name, df in sheets.items())
    if suffix == ".csv":
        return pd.read_csv(path).to_csv(index=False)
    if suffix in (".mp3", ".wav", ".png", ".jpg", ".jpeg"):
        return f"Cannot read {suffix} files: this agent has no audio or vision support."
    return path.read_text(encoding="utf-8", errors="replace")


ANSWER_INSTRUCTIONS = """
You are answering a question from the GAIA benchmark. Your answer is graded by EXACT MATCH.
- Call final_answer with ONLY the answer: a number, a few words, or a comma separated list.
- No explanations, no units unless asked, no "FINAL ANSWER:" prefix, no trailing period.
- Numbers: digits only, no thousands separators. Strings: no articles or abbreviations unless asked.
- Use web search and visit pages to verify facts instead of guessing.
"""


class GaiaAgent:
    def __init__(self):
        model = build_model()
        self.agent = CodeAgent(
            model=model,
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), read_file],
            additional_authorized_imports=["pandas", "numpy", "math", "re", "json", "datetime"],
            code_block_tags="markdown",
            verbosity_level=1,
            max_steps=10,
        )

    def __call__(self, question: str, file_path: Optional[str] = None) -> str:
        prompt = f"{ANSWER_INSTRUCTIONS}\nQuestion: {question}"
        if file_path:
            prompt += f"\n\nThe task has an attached file at: {file_path} (use read_file to read it)."
        answer = self.agent.run(prompt)
        return clean_answer(answer)


def clean_answer(answer) -> str:
    text = str(answer).strip()
    text = re.sub(r"^\s*final answer\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(".").strip()
