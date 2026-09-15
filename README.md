# 🎞️ Rundown — AI Video & Meeting Assistant

Turn any recording — a YouTube video or a local audio/video file — into a structured rundown: a title, a summary, action items, key decisions, and open questions. Then chat with the recording itself to ask follow-up questions, powered by a Retrieval-Augmented Generation (RAG) pipeline over the transcript.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/orchestration-LangChain-1C3C3C)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036)

---

## What it does

Point it at a recording — a YouTube link or an uploaded audio/video file — and it will:

1. Extract and chunk the audio
2. Transcribe it (English via a local Whisper model, or Hinglish via Sarvam AI's speech-to-text-translate API)
3. Generate a short title and a map-reduce style summary
4. Pull out action items (with owner and deadline), key decisions, and open questions
5. Build a local vector store over the transcript so you can chat with it and ask follow-up questions

It ships as both a **Streamlit web app** and a **command-line tool**, sharing the same pipeline.

## Features

- 🎥 **Two input sources** — paste a YouTube URL, or upload a local file (`mp4`, `mov`, `mkv`, `avi`, `webm`, `mp3`, `wav`, `m4a`)
- ✂️ **Automatic chunking** of long recordings before transcription
- 🗣️ **Dual transcription engines**
  - `english` → local OpenAI Whisper model (offline, no API key)
  - `hinglish` → [Sarvam AI](https://www.sarvam.ai)'s speech-to-text-translate API (transcribes Hindi/Hinglish speech and translates it to English)
- 🧠 **LLM-powered analysis** via Groq (`openai/gpt-oss-20b`, orchestrated with LangChain LCEL):
  - Auto-generated title
  - Per-chunk summaries combined into one final bullet-point summary
  - Action items with owner & deadline
  - Key decisions
  - Open questions / follow-ups
- 💬 **Chat with the recording** — a RAG chain over a local Chroma vector store (HuggingFace `all-MiniLM-L6-v2` embeddings) answers questions using only the transcript as context
- 🖥️ **Streamlit UI ("Rundown")** — staged progress view, tabs for summary/action items/decisions/questions/transcript, transcript search & highlighting, and downloadable summary (`.md`) and transcript (`.txt`)
- ⌨️ **CLI mode** (`main.py`) — runs the same pipeline and drops you into an interactive terminal chat once processing finishes

## How it works

```
YouTube URL / local file
        │
        ▼
utils/audio_processor.py   → download (yt-dlp) or convert (pydub) to WAV, split into chunks
        │
        ▼
core/transcriber.py        → Whisper (english) or Sarvam AI (hinglish) per chunk
        │
        ▼
core/summerise.py          → title + map-reduce summary
core/extractor.py          → action items, key decisions, open questions
        │
        ▼
core/vector_store.py       → chunk + embed transcript into a local Chroma store
core/rag_engine.py         → retrieval chain for follow-up chat
```

## Tech stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (LCEL), `langchain-groq` |
| LLM | Groq — `openai/gpt-oss-20b` |
| Speech-to-text | OpenAI Whisper (local), Sarvam AI (Hinglish) |
| Vector store / embeddings | Chroma, HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Audio pipeline | `yt-dlp`, `pydub`, `static-ffmpeg` |

## Project structure

```
AI-VIDEO-ASSISTANT/
├── app.py                     # Streamlit UI ("Rundown")
├── main.py                    # CLI entry point
├── test.py                    # Ad-hoc pipeline test script
├── core/
│   ├── transcriber.py         # Whisper / Sarvam transcription
│   ├── summerise.py           # Title + summary generation
│   ├── extractor.py           # Action items / decisions / questions
│   ├── rag_engine.py          # RAG chain for chat
│   └── vector_store.py        # Chroma vector store
├── utils/
│   └── audio_processor.py     # YouTube download, format conversion, chunking
├── requirements.txt
├── pyproject.toml / uv.lock
└── .python-version
```

## Getting started

### Prerequisites

- Python 3.11
- FFmpeg (installed automatically via `static-ffmpeg`)
- A [Groq API key](https://console.groq.com/keys) — required for every LLM step (summary, extraction, chat)
- A [Sarvam AI API key](https://www.sarvam.ai) — only needed if you plan to use `hinglish` transcription

### Installation

```bash
git clone https://github.com/Rounaksenskr/AI-VIDEO-ASSISTANT.git
cd AI-VIDEO-ASSISTANT

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install langchain-groq langchain-chroma static-ffmpeg   # used by the code; add if pip complains they're missing
```

This repo also includes a `pyproject.toml` / `uv.lock`, so if you use [uv](https://github.com/astral-sh/uv) you can run `uv sync` instead.

### Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key      # only needed for hinglish mode
WHISPER_MODEL=small                     # optional, defaults to "small"
SARVAM_STT_MODEL=saaras:v2.5            # optional
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Paste a YouTube URL or upload a recording in the sidebar, pick a language, and click **Process recording**.

### Run the CLI

```bash
python main.py
```

You'll be prompted for a YouTube URL or file path and a language. Once processing finishes you're dropped into an interactive chat loop over the transcript (type `exit` to quit).

## Language support

| Language | Engine | Notes |
|---|---|---|
| `english` | OpenAI Whisper (local) | Runs on your machine, no API key needed |
| `hinglish` | Sarvam AI `speech-to-text-translate` | Audio is split into 25s pieces to fit Sarvam's 30-second sync API limit, then transcribed and translated to English |

## Notes

- The vector store is persisted locally to `vector_db/` and downloaded/converted audio to a local folder — both are regenerated per run and excluded via `.gitignore`.
- `requirements.txt` includes a few extra libraries (e.g. PDF export tooling) that aren't wired into the app yet; the packages listed above are what's actually needed to run it today.

## Roadmap

- [ ] PDF export for meeting summaries
- [ ] Broader language support beyond English/Hinglish
- [ ] Persistent history across sessions

## Author

Built by [Rounak](https://github.com/Rounaksenskr).
