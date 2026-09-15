"""
Rundown — a Streamlit front end for the AI video/meeting assistant pipeline.

Wraps the same functions used in main.py (process_input, transcribe_all,
generate_title, summarize, extract_action_items, extract_key_decisions,
extract_questions, build_rag_chain, ask_question) with a staged-progress
UI, a results view, and a chat tab backed by the RAG chain.

Run with:  streamlit run app.py
"""

import html
import os
import re
import tempfile
import traceback

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summerise import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Rundown — AI video & meeting assistant",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #12151b;
    --bg-panel: #1a1f29;
    --bg-panel-alt: #20262f;
    --border: #2a313d;
    --text-primary: #eceef1;
    --text-secondary: #9aa3b2;
    --accent-amber: #e8a33d;
    --accent-teal: #4fb6a8;
    --accent-rust: #c2624a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.5rem;
    max-width: 1100px;
}

#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary); }
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label, [data-testid="stSidebar"] .stFileUploader label {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.brand-mark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
}
.brand-tagline {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.4;
    margin-bottom: 1.4rem;
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-panel-alt) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
.stRadio [role="radiogroup"] label {
    background-color: var(--bg-panel-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    margin-right: 0.4rem;
}

/* Buttons */
.stButton > button {
    background-color: var(--accent-amber);
    color: var(--bg);
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1rem;
    width: 100%;
    transition: filter 0.15s ease;
}
.stButton > button:hover {
    filter: brightness(1.08);
    color: var(--bg);
}
.stButton > button[kind="secondary"] {
    background-color: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.stDownloadButton > button {
    background-color: var(--bg-panel-alt);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* Tabs -> pill segmented control */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    height: 2.4rem;
    background-color: transparent;
    border-radius: 6px 6px 0 0;
    color: var(--text-secondary);
    font-weight: 500;
    padding: 0 1rem;
}
.stTabs [aria-selected="true"] {
    background-color: var(--bg-panel);
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-amber);
}

/* Hero */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    line-height: 1.25;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}
.hero-meta {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
}

/* Empty state */
.empty-lede {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.9rem;
    line-height: 1.3;
    max-width: 34rem;
    margin-bottom: 0.6rem;
}
.empty-sub {
    color: var(--text-secondary);
    max-width: 34rem;
    line-height: 1.55;
    margin-bottom: 2rem;
}
.step-row {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    padding: 0.7rem 0;
    border-top: 1px solid var(--border);
    max-width: 34rem;
}
.step-row:last-child { border-bottom: 1px solid var(--border); }
.step-num {
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-amber);
    font-size: 0.95rem;
    padding-top: 0.1rem;
}
.step-body b { color: var(--text-primary); }
.step-body { color: var(--text-secondary); font-size: 0.92rem; line-height: 1.5; }

/* Section panels */
.panel {
    border-left: 3px solid var(--border);
    padding: 0.2rem 0 0.2rem 1rem;
    margin-bottom: 0.4rem;
}
.panel-amber { border-left-color: var(--accent-amber); }
.panel-teal { border-left-color: var(--accent-teal); }
.panel-rust { border-left-color: var(--accent-rust); }

.item-row {
    display: flex;
    gap: 0.6rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    line-height: 1.45;
}
.item-row:last-child { border-bottom: none; }
.item-icon {
    color: var(--accent-amber);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}
.empty-note {
    color: var(--text-secondary);
    font-style: normal;
    padding: 0.6rem 0;
}

/* Transcript */
.transcript-box {
    background-color: var(--bg-panel-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    max-height: 480px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    white-space: pre-wrap;
    color: var(--text-primary);
}
.transcript-box::-webkit-scrollbar { width: 8px; }
.transcript-box::-webkit-scrollbar-thumb { background-color: var(--border); border-radius: 4px; }
mark {
    background-color: var(--accent-amber);
    color: var(--bg);
    border-radius: 2px;
    padding: 0 2px;
}

[data-testid="stChatMessage"] {
    background-color: var(--bg-panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

LANGUAGE_OPTIONS = ["english", "hinglish"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def to_list(value):
    """Normalize an extractor's output (string or list) into a clean list of items."""
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).splitlines()
    cleaned = []
    for item in items:
        item = str(item).strip()
        item = re.sub(r"^[\-\*\u2022\d]+[\.\)]?\s*", "", item)
        if item:
            cleaned.append(item)
    return cleaned


def save_upload_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def render_items(items, icon):
    if not items:
        st.markdown("<div class='empty-note'>Nothing was found here for this recording.</div>", unsafe_allow_html=True)
        return
    rows = "".join(
        f"<div class='item-row'><span class='item-icon'>{icon}</span><span>{html.escape(item)}</span></div>"
        for item in items
    )
    st.markdown(rows, unsafe_allow_html=True)


def run_pipeline_with_progress(source, language):
    with st.status("Processing your recording…", expanded=True) as status:
        status.write("Extracting audio…")
        chunks = process_input(source)

        status.write("Transcribing audio — this can take a while for longer recordings…")
        transcript = transcribe_all(chunks, language=language)

        status.write("Naming the recording…")
        title = generate_title(transcript)

        status.write("Summarizing the key points…")
        summary = summarize(transcript)

        status.write("Pulling out action items…")
        action_items = extract_action_items(transcript)

        status.write("Finding key decisions…")
        decisions = extract_key_decisions(transcript)

        status.write("Collecting open questions…")
        questions = extract_questions(transcript)

        status.write("Indexing the transcript for chat…")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Your rundown is ready", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": to_list(action_items),
        "key_decisions": to_list(decisions),
        "open_questions": to_list(questions),
        "rag_chain": rag_chain,
    }


def reset_session():
    temp_path = st.session_state.get("temp_source_path")
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    for key in ("result", "chat_history", "temp_source_path", "source_label"):
        st.session_state.pop(key, None)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<div class='brand-mark'>🎞️ Rundown</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='brand-tagline'>Turn a recording into a rundown you can act on.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("result"):
        st.caption("Currently showing")
        st.write(f"**{st.session_state['result']['title']}**")
        if st.button("Start a new recording", type="secondary"):
            reset_session()
            st.rerun()
        st.divider()
    else:
        st.markdown("**New recording**")

        source_kind = st.radio("Source", ["YouTube URL", "Upload file"], label_visibility="visible")

        source = None
        source_label = None
        if source_kind == "YouTube URL":
            url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…")
            if url.strip():
                source = url.strip()
                source_label = "a YouTube link"
        else:
            uploaded = st.file_uploader(
                "Audio or video file",
                type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a"],
            )
            if uploaded is not None:
                source_label = f"the uploaded file \"{uploaded.name}\""

        language = st.selectbox("Language", LANGUAGE_OPTIONS, index=0)

        process_clicked = st.button("Process recording", type="primary")

        if process_clicked:
            if source_kind == "Upload file" and uploaded is None:
                st.error("Add a file before processing.")
            elif source_kind == "YouTube URL" and not source:
                st.error("Add a YouTube URL before processing.")
            else:
                try:
                    if source_kind == "Upload file":
                        source = save_upload_to_temp(uploaded)
                        st.session_state["temp_source_path"] = source
                    result = run_pipeline_with_progress(source, language)
                    st.session_state["result"] = result
                    st.session_state["source_label"] = source_label
                    st.session_state["chat_history"] = []
                    st.rerun()
                except Exception as exc:
                    st.error(f"We couldn't process that recording: {exc}")
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

result = st.session_state.get("result")

if not result:
    st.markdown(
        "<div class='empty-lede'>Turn a recording into a rundown you can act on.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='empty-sub'>Drop in a YouTube link or a local recording, and get a title, "
        "summary, action items, decisions, and open questions — plus a chat window to ask "
        "follow-up questions about what was said.</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Add a recording", "Paste a YouTube link or upload an audio or video file in the sidebar."),
        ("2", "We do the listening", "The recording is transcribed, then summarized and combed for action items, decisions, and questions."),
        ("3", "Read or ask", "Skim the rundown in the tabs below, or chat with the recording directly."),
    ]
    rows = "".join(
        f"<div class='step-row'><span class='step-num'>{num}</span>"
        f"<span class='step-body'><b>{title}</b><br>{desc}</span></div>"
        for num, title, desc in steps
    )
    st.markdown(rows, unsafe_allow_html=True)

else:
    transcript = result["transcript"]
    word_count = len(transcript.split())
    source_label = st.session_state.get("source_label") or "a recording"

    st.markdown(f"<div class='hero-title'>{html.escape(result['title'])}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='hero-meta'>Transcribed from {source_label} — {word_count:,} words total.</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Summary", "Action items", "Decisions", "Questions", "Transcript", "Chat"])

    with tabs[0]:
        st.markdown("<div class='panel panel-amber'></div>", unsafe_allow_html=True)
        st.markdown(result["summary"])
        st.download_button(
            "Download summary (.md)",
            data=result["summary"],
            file_name=f"{result['title']}_summary.md",
        )

    with tabs[1]:
        st.markdown("<div class='panel panel-amber'></div>", unsafe_allow_html=True)
        render_items(result["action_items"], "☐")

    with tabs[2]:
        st.markdown("<div class='panel panel-teal'></div>", unsafe_allow_html=True)
        render_items(result["key_decisions"], "◆")

    with tabs[3]:
        st.markdown("<div class='panel panel-rust'></div>", unsafe_allow_html=True)
        render_items(result["open_questions"], "?")

    with tabs[4]:
        search_term = st.text_input("Search the transcript", placeholder="Search for a word or phrase…")
        display_text = html.escape(transcript)
        if search_term.strip():
            pattern = re.compile(re.escape(html.escape(search_term.strip())), re.IGNORECASE)
            display_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", display_text)
        st.markdown(f"<div class='transcript-box'>{display_text}</div>", unsafe_allow_html=True)
        st.download_button(
            "Download transcript (.txt)",
            data=transcript,
            file_name=f"{result['title']}_transcript.txt",
        )

    with tabs[5]:
        st.session_state.setdefault("chat_history", [])
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask something about this recording…")
        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as exc:
                        answer = f"I couldn't answer that: {exc}"
                st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})