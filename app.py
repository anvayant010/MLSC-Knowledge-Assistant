"""Streamlit chat UI for the MLSC Knowledge Assistant.

Run from the project root with the venv active:
    streamlit run app.py
"""

import os

import streamlit as st

from src.config import KNOWLEDGE_BASE_DIR
from src.rag import RagPipeline

EXAMPLE_QUESTIONS = [
    "What technical domains exist in MLSC?",
    "How does someone become a coordinator?",
    "What stages happen during an MLSC hackathon?",
    "Is there a registration fee for hackathons?",
]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 1.75rem 2.25rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
}
.hero-banner h1 {
    margin: 0;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
}
.hero-banner p {
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.9);
    line-height: 1.4;
}

.source-pill {
    display: inline-block;
    background: rgba(129, 140, 248, 0.15);
    color: #A5B4FC;
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.35rem 0.2rem 0;
}

.example-label {
    font-size: 0.85rem;
    color: #9CA3AF;
    margin: 0.25rem 0 0.6rem 0;
    font-weight: 500;
}
</style>
"""

st.set_page_config(page_title="MLSC Knowledge Assistant", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base and embedding model...")
def load_pipeline() -> RagPipeline:
    return RagPipeline()


st.markdown(
    """
    <div class="hero-banner">
        <h1>MLSC Knowledge Assistant</h1>
        <p>Ask anything about MLSC domains, leadership, membership, hackathons or community
        guidelines. Every answer is grounded in the official knowledge base, with sources cited
        and unsupported questions flagged honestly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(
        "Could not start the assistant. If you're using the Gemini backend, check that "
        "GEMINI_API_KEY is set in .env. If you're using Ollama, check that it's running "
        f"locally.\n\nDetails: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
    st.markdown(pills, unsafe_allow_html=True)


st.markdown('<p class="example-label">Try asking:</p>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, eq in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(eq, use_container_width=True, key=f"example_{eq}"):
        st.session_state.pending_question = eq

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask about MLSC domains, leadership, membership, hackathons...")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        def on_retry(message: str) -> None:
            status_placeholder.info(message)

        try:
            with st.spinner("Thinking..."):
                result = pipeline.answer(question, on_retry=on_retry)
        except Exception as e:
            status_placeholder.empty()
            provider = os.environ.get("LLM_PROVIDER", "gemini")
            hint = (
                "Make sure Ollama is running locally (`ollama serve`)."
                if provider == "ollama"
                else "Check your GEMINI_API_KEY and network connection."
            )
            st.error(f"Something went wrong while generating the answer. {hint}\n\nDetails: {e}")
            result = None

        if result is not None:
            status_placeholder.empty()
            st.markdown(result.answer)
            render_sources(result.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )

with st.sidebar:
    st.markdown("### About this system")
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    st.markdown(f"**LLM backend:** `{provider}`")

    st.markdown("**Knowledge base documents:**")
    for doc in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        st.markdown(f"- {doc.name}")

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
