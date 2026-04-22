# KOBRA

> A local voice AI assistant, multi-agent system, and personal OS controller — built to run entirely on your own machine.

KOBRA is not a wrapper around a chatbot. It's a cognitive architecture: wake word detection, real-time speech recognition, multi-agent task decomposition, parallel tool execution, persistent memory, and natural voice output — all running locally on Windows with no mandatory cloud dependency.

Think JARVIS. Built from scratch. In Python.

---

## What It Does

- **Listens continuously** via wake word — say "Hey KOBRA", it wakes up
- **Understands natural language** commands, questions, and multi-step instructions
- **Decomposes complex tasks** into parallel subtasks routed to specialized agents
- **Controls your OS** — opens apps, manages windows, types, clicks, runs commands
- **Searches the web** and synthesizes results into spoken answers
- **Plays media** — YouTube, Spotify, local audio
- **Manages your calendar and email** via Google APIs
- **Sends WhatsApp messages** via browser automation
- **Remembers everything** — conversations, facts, preferences, project context — across sessions
- **Knows your files** — indexes your projects and documents via local RAG, answers questions about your own work
- **Proactive intelligence** — morning briefings, calendar alerts, build failure notifications
- **Adapts to you** — learns your vocabulary, response preferences, usage patterns, and routing corrections over time

---

## Architecture

```
[Wake Word — Porcupine]
        ↓
[STT — faster-whisper]
        ↓
[NeuralBrain]
    ├── Planner         — decompose into steps
    ├── Memory Router   — episodic + semantic + procedural + perceptual
    ├── Guardrails      — safety checks before execution
    └── Executor Loop
            ├── SystemAgent     — apps, shell, system info
            ├── WebAgent        — search, browser
            ├── DevAgent        — files, VS Code, scaffolding
            ├── MediaAgent      — YouTube, Spotify, audio
            ├── MemoryAgent     — save, recall
            ├── KnowledgeAgent  — RAG over local files
            ├── ResearchAgent   — deep web research + Playwright
            └── ScreenAgent     — vision-based OS control
        ↓
[Reflection + Retry]
        ↓
[Aggregator — synthesize results]
        ↓
[TTS — Kokoro / edge-tts]
        ↓
[Learning System — update memory]
```

---

## Stack

| Component | Technology |
|---|---|
| Wake word | Porcupine (Picovoice) |
| Speech-to-text | faster-whisper |
| LLM — tool calling | Groq (`llama-3.3-70b-versatile`) |
| LLM — fast path | Groq (`llama3-8b-8192`) |
| Text-to-speech | Kokoro TTS / edge-tts fallback |
| Audio playback | ffmpeg / sounddevice |
| Memory — conversations | SQLite (episodic) |
| Memory — facts | SQLite (semantic) |
| Memory — RAG | ChromaDB + nomic-embed-text (Ollama) |
| Browser automation | Playwright |
| Email + Calendar | Google APIs |
| Offline LLM fallback | Ollama (`qwen2.5:3b`) |
| OS automation | pyautogui + pywinauto |
| Backend language | Python 3.11+ |
| Platform | Windows (WSL2 compatible) |

---

## Project Structure

```
kobra/
├── main.py                  # entry point and main loop
├── config.py                # all configuration and constants
├── brain.py                 # Groq LLM integration, tool calling, intent routing
├── listener.py              # wake word + STT pipeline
├── speaker.py               # TTS synthesis and playback
│
├── neural/
│   ├── planner.py           # task decomposition, enrichment
│   ├── executor.py          # step-by-step execution loop, working memory
│   ├── reflection.py        # per-step success checking, confidence scoring
│   ├── guardrails.py        # safety checks, loop detection, command whitelisting
│   ├── aggregator.py        # result merging, response synthesis
│   ├── learning.py          # 5-dimension learning system
│   └── journal.py           # append-only execution log
│
├── memory/
│   ├── router.py            # unified memory query interface
│   ├── episodic.py          # conversation history
│   ├── semantic.py          # facts, preferences, vocabulary
│   ├── procedural.py        # routing strategies, tool success rates
│   └── perceptual.py        # RAG + live API data
│
├── agents/
│   ├── base_agent.py
│   ├── system_agent.py
│   ├── web_agent.py
│   ├── dev_agent.py
│   ├── media_agent.py
│   ├── memory_agent.py
│   ├── conversation_agent.py
│   ├── knowledge_agent.py
│   ├── research_agent.py
│   └── screen_agent.py
│
├── rag/
│   ├── indexer.py           # file reading and chunking
│   ├── embedder.py          # nomic-embed-text via Ollama
│   ├── store.py             # ChromaDB operations
│   ├── watcher.py           # watchdog file watcher
│   └── retriever.py         # semantic search and reranking
│
├── proactive/
│   ├── briefing.py          # morning briefing engine
│   ├── scanners.py          # calendar, email, project, GitHub scanners
│   └── watcher.py           # threshold-based continuous monitoring
│
└── tools/
    ├── system.py            # open_app, run_command, get_system_info
    ├── web.py               # web_search, open_url
    ├── media.py             # play_youtube, play_media
    ├── dev.py               # create_file, scaffold_project, open_vscode
    ├── window.py            # window management, focus modes
    ├── screen.py            # screenshot, vision-based clicking
    └── browser.py           # Playwright automation
```

---

## Setup

### Prerequisites

- Windows 10/11 (WSL2 optional)
- Python 3.11+
- NVIDIA GPU recommended (RTX series) — CPU fallback supported
- ffmpeg in system PATH
- Ollama installed and running

### Installation

```bash
git clone https://github.com/KobraKob/kobra.git
cd kobra
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
PORCUPINE_ACCESS_KEY=your_porcupine_key
HUME_API_KEY=your_hume_key_optional
ELEVENLABS_API_KEY=your_elevenlabs_key_optional
```

### Configuration

Edit `config.py` to set your watched folders for RAG indexing:

```python
WATCHED_FOLDERS = [
    "C:/Users/YourName/Projects",
    "C:/Users/YourName/Documents",
]
```

### Run

```bash
python main.py
```

Say **"Hey KOBRA"** to activate.

---

## Example Commands

```
"Hey KOBRA, what's on my calendar today?"
"Hey KOBRA, search for the latest news on AI agents and open the top result"
"Hey KOBRA, scaffold a FastAPI project called dashboard and open it in VS Code"
"Hey KOBRA, play some lofi music"
"Hey KOBRA, send an email to John that the meeting is at 6 PM"
"Hey KOBRA, what was I working on yesterday?"
"Hey KOBRA, remember that my current project is KOBRA"
"Hey KOBRA, what do my project files say about the memory architecture?"
"Hey KOBRA, focus mode"
"Hey KOBRA, what's on my screen?"
```

---

## Key Design Decisions

**Tool calling over intent parsing** — Groq decides which tool to call and with what arguments. No regex, no keyword matching, no brittle pattern mapping.

**Parallel execution** — independent subtasks run concurrently via `ThreadPoolExecutor`. One command can trigger web search, file creation, and app launch simultaneously.

**Four-layer memory** — episodic (what happened), semantic (what's true), procedural (how to do things), perceptual (what exists now). Unified query interface across all layers via `MemoryRouter`.

**Local-first** — STT, embeddings, and fallback LLM all run locally. Groq is the only mandatory cloud dependency and has a generous free tier.

**Cooperative abort** — every agent checks an abort flag before and between tool calls. "Stop" or "cancel" during execution cleanly halts the pipeline without killing processes or corrupting state.

---

## Roadmap

- [x] Wake word detection
- [x] Speech-to-text pipeline
- [x] Groq LLM integration with tool calling
- [x] Multi-agent orchestration
- [x] Parallel tool execution
- [x] Persistent memory (episodic + semantic)
- [x] RAG over local files
- [x] Gmail + Google Calendar integration
- [x] WhatsApp automation
- [x] Window management + Focus Mode
- [ ] Parakeet TDT v3 STT upgrade
- [ ] Neural cognitive architecture (Planner → Executor → Reflect → Retry)
- [ ] Morning briefing engine
- [ ] Emotion detection
- [ ] Fine-tune STT on personal voice
- [ ] Vision-based screen understanding

---

## Author

**Balavanth** — Systems Engineer at TCS, building toward AI Engineering.

- GitHub: [KobraKob](https://github.com/KobraKob)
- LinkedIn: [balavanth](https://linkedin.com/in/balavanth)

---

## License

MIT License — use it, modify it, build on it.

---

> *"Not a wrapper. Not a demo. A real system built from scratch."*
