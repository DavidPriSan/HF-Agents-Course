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
    WikipediaSearchTool,
    tool,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")


# Model selection (AGENT_MODEL env var):
#   "ollama"            -> local Ollama (default, OLLAMA_MODEL picks the model)
#   "hf:<model_id>"     -> HF Inference Providers (needs HF_TOKEN)
#   anything else       -> LiteLLM model id, e.g. "gemini/gemini-2.5-flash" (needs GEMINI_API_KEY);
#                          a "litellm:" prefix is also accepted
def build_model():
    choice = os.getenv("AGENT_MODEL", "ollama")
    print(f"Using model: {choice}")
    if choice.startswith("hf:"):
        return InferenceClientModel(model_id=choice[3:], token=os.getenv("HF_TOKEN"))
    if choice != "ollama":
        # Stay under free-tier per-minute limits (AGENT_RPM, e.g. 15 for Gemini Flash Lite)
        rpm = float(os.getenv("AGENT_RPM", "0")) or None
        return LiteLLMModel(model_id=choice.removeprefix("litellm:"), requests_per_minute=rpm)
    return LiteLLMModel(
        model_id=os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:7b"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        num_ctx=8192,
    )


def model_has_vision() -> bool:
    return os.getenv("AGENT_MODEL", "ollama") != "ollama"


@tool
def read_file(file_path: str) -> str:
    """
    Reads a local file attached to the task and returns its content as text.
    Supports spreadsheets (.xlsx, .xls, .csv), audio (.mp3, .wav: returns a transcription)
    and text files (.py, .txt, .json, .md, ...).

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
    if suffix in AUDIO_EXTENSIONS:
        return transcribe_audio(str(path))
    if suffix in IMAGE_EXTENSIONS:
        return "This is an image. It was attached to the task as an image if the model supports vision."
    return path.read_text(encoding="utf-8", errors="replace")


_whisper = None


def transcribe_audio(file_path: str) -> str:
    # Runs on CPU so it doesn't compete with Ollama for the 4 GB of VRAM
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(file_path)
    return " ".join(segment.text.strip() for segment in segments)


@tool
def youtube_transcript(url: str) -> str:
    """
    Returns the transcript (spoken words / subtitles) of a YouTube video.
    It cannot describe what is shown on screen, only what is said.

    Args:
        url: The YouTube video URL.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    if not match:
        return f"Could not find a video id in {url}"
    try:
        transcript = YouTubeTranscriptApi().fetch(match.group(1))
    except Exception as e:
        return f"No transcript available: {e}"
    return " ".join(snippet.text for snippet in transcript.snippets)


@tool
def reverse_text(text: str) -> str:
    """
    Reverses a string character by character. Useful when a question is written backwards.

    Args:
        text: The text to reverse.
    """
    return text[::-1]


ANSWER_INSTRUCTIONS = """
You are answering a question from the GAIA benchmark. Your answer is graded by EXACT MATCH.
- Call final_answer with ONLY the answer: a number, a few words, or a comma separated list.
- No explanations, no units unless asked, no "FINAL ANSWER:" prefix, no trailing period.
- Numbers: digits only, no thousands separators. Strings: no articles or abbreviations unless asked.
- Comma separated lists: use ", " between items.
- Never pass a whole variable or search result to final_answer: extract the exact value first.
- Prefer wikipedia_search for encyclopedic facts, then web_search + visit_webpage to verify.
- If the question looks unreadable, it may be written backwards: try reverse_text.
"""


def check_short_answer(final_answer, memory, agent=None) -> bool:
    # Reject dumps of lists/search results so the agent extracts the real value and retries
    text = str(final_answer).strip()
    if not text:
        raise ValueError("The final answer is empty.")
    if len(text) > 200 or text.startswith("[") or "\n" in text:
        raise ValueError(
            "The final answer must be ONLY the short answer (a number, a few words or a comma "
            "separated list), not a list dump or an explanation. Extract the exact value and call final_answer again."
        )
    return True


class GaiaAgent:
    def __init__(self):
        self.agent = CodeAgent(
            model=build_model(),
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                WikipediaSearchTool(user_agent="HF-Agents-Course (DavidPriSan)"),
                youtube_transcript,
                reverse_text,
                read_file,
            ],
            additional_authorized_imports=["pandas", "numpy", "math", "re", "json", "datetime", "collections", "itertools"],
            code_block_tags="markdown",
            final_answer_checks=[check_short_answer],
            verbosity_level=1,
            max_steps=12,
        )

    def __call__(self, question: str, file_path: Optional[str] = None) -> str:
        prompt = f"{ANSWER_INSTRUCTIONS}\nQuestion: {question}"
        images = None
        if file_path:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".py":
                # Small and decisive: inline the code so the model doesn't have to fetch it
                code = Path(file_path).read_text(encoding="utf-8", errors="replace")
                prompt += f"\n\nAttached Python file ({file_path}):\n```python\n{code}\n```"
            elif suffix in IMAGE_EXTENSIONS and model_has_vision():
                from PIL import Image

                images = [Image.open(file_path)]
                prompt += "\n\nThe attached image is provided to you."
            else:
                prompt += f"\n\nThe task has an attached file at: {file_path} (use read_file to read it)."
        answer = self.agent.run(prompt, images=images)
        return clean_answer(answer)


def clean_answer(answer) -> str:
    text = str(answer).strip()
    text = re.sub(r"^\s*final answer\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(".").strip()
