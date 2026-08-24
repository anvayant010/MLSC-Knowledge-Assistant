# MLSC Knowledge Assistant

A retrieval-augmented question-answering system for the MLSC knowledge base. Ask a
question in natural language, get an answer grounded in the official MLSC documents
with sources cited, and get an honest "I don't know" when the answer isn't in the
knowledge base — instead of a hallucinated one.

Built for the MLSC AIML Domain Lead recruitment challenge.

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)

See [APPROACH.md](APPROACH.md) for the approach and major technical decisions.

## Quick start

### 1. Set up the environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.11+ (tested on 3.13).

### 2. Configure an LLM backend

Copy `.env.example` to `.env` and fill in one of the two supported backends:

```
LLM_PROVIDER=gemini            # or "ollama"
GEMINI_API_KEY=your_key_here   # from https://aistudio.google.com
GEMINI_MODEL=gemini-flash-latest

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

- **Gemini** (default, cloud): get a free API key at Google AI Studio. Free tier is
  rate-limited (as low as 5 requests/minute depending on model) — the app retries
  automatically with backoff if you hit it.
- **Ollama** (local, no API key): install [Ollama](https://ollama.com), run
  `ollama pull llama3`, and make sure `ollama serve` (or the desktop app) is running.

### 3. Build the knowledge base index

```bash
python -m src.ingest
```

This reads every `.txt` file in `AboutMLSC/`, embeds each document, and stores the
result in a local SQLite database (`knowledge_base.db`, gitignored — regenerate it
with this command any time the source documents change).

### 4. Run the app

```bash
streamlit run app.py
```

Opens a chat UI at `http://localhost:8501`. Ask a question, get an answer with the
source document(s) cited below it.

### 5. Run the evaluation

```bash
python -m eval.evaluate
```

Runs the 18-question eval set against the pipeline and writes `eval/results.md` /
`eval/results.json`. Defaults to the Ollama backend to avoid Gemini rate limits during
a batch run of many calls; override with `LLM_PROVIDER=gemini python -m eval.evaluate`
to evaluate the cloud backend instead — it's the same pipeline code either way, only
the model swaps.

## How it works

```
question → embed → retrieve top matching document(s) → build grounded prompt
         → LLM generates answer → extract cited sources → return to user
```

1. **Ingestion** (`src/ingest.py`): each of the 6 knowledge-base files is embedded as
   a single whole-document chunk using `sentence-transformers` (`all-MiniLM-L6-v2`,
   local, no API key) and stored in SQLite.
2. **Retrieval** (`src/rag.py`): the question is embedded and compared against all
   documents via cosine similarity (`src/store.py`, plain NumPy — no vector database).
   The top few matches are kept, trimmed by a relative-score margin so a document that
   clearly isn't relevant doesn't get pulled in just to fill a fixed top-k slot.
3. **Unanswerable detection**: if the best match's similarity score is below a
   threshold, or if the LLM itself determines the retrieved context doesn't answer
   the question, the app returns an explicit "not in the knowledge base" message
   instead of guessing.
4. **Generation**: a system prompt instructs the LLM to answer only from the
   retrieved context, synthesize across multiple documents when needed, and state
   plainly when something isn't covered.
5. **Sources**: the set of documents actually retrieved for the answer is shown to
   the user, satisfying the requirement to cite sources — including for questions
   that need more than one document.

## Evaluation

Metrics, computed by `eval/evaluate.py` against the 18-question eval set (see
`eval/results.md` for the full per-question breakdown):

| Metric | Score | How it's computed |
|---|---|---|
| Context Precision | 0.85 | Of retrieved documents, fraction that were actually relevant (labeled ground truth), averaged over answerable questions |
| Context Recall | 0.93 | Of the labeled-relevant documents, fraction actually retrieved |
| Answer Relevancy | 0.99 | LLM-judge score (1–5, normalized) rating whether the answer directly addresses the question |
| Faithfulness / Groundedness | 0.99 | LLM-judge score rating whether every claim in the answer is supported by the retrieved context (checks for hallucination) |
| Unanswerable Detection Accuracy | 1.00 | Fraction of questions where the system correctly recognized an answerable vs. unanswerable question |

Context Precision/Recall are computed directly (no LLM needed) from the overlap
between retrieved source files and hand-labeled expected sources. Answer Relevancy
and Faithfulness require an LLM judge, since "does this answer make sense" and "is
this claim actually supported by the text" aren't computable from string matching —
the eval harness uses the same pluggable `LLMClient`, defaulting to Ollama to keep a
~50-call batch run fast and rate-limit-free.

## Project structure

```
AboutMLSC/              knowledge base (6 .txt files, source of truth)
src/
  config.py              paths, embedding model, retrieval tuning constants
  ingest.py               chunk + embed + store the knowledge base
  store.py                SQLite storage + NumPy cosine-similarity search
  llm.py                  pluggable LLM client (Gemini / Ollama) with retry logic
  rag.py                  retrieval + prompting + generation pipeline
eval/
  eval_set.json            18 hand-written questions across 5 categories
  evaluate.py               evaluation harness
  results.md / results.json  latest evaluation report
app.py                    Streamlit chat UI
requirements.txt
.env.example
```

## Known limitations

- **Generic/broad queries can under-retrieve.** MiniLM (a small, local embedding
  model) sometimes struggles to separate short, topically-similar documents on
  vague questions (e.g. "What is MLSC and what does it focus on?" scored the correct
  document slightly below two others). A larger embedding model would likely narrow
  this gap, at the cost of no longer being free/local.
- **Ollama's local model (`llama3`) is noticeably less thorough than Gemini** on
  broad, open-ended questions — it initially tended to answer with only a fragment
  of the available context rather than synthesizing the whole document; the system
  prompt was tightened to explicitly require full coverage for open-ended questions,
  which improved this but a stronger model still gives more consistently complete
  answers.
- The knowledge base itself doesn't name specific individuals (e.g. who currently
  holds a given leadership role), so questions asking for names are correctly
  answered as unavailable rather than fabricated — this is a knowledge base
  limitation, not a retrieval failure.
