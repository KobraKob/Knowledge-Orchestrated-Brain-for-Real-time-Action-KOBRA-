"""
orchestrator.py — Master controller for KOBRA v3.

Pipeline for every user command:
  1. Decompose transcript → list[Task]  (Groq JSON call, fast model)
  2. TaskRouter validates + enriches    (or rule-based fallback)
  3. TaskQueue executes tasks           (parallel where safe, sequential otherwise)
  4. Synthesizer generates response     (one coherent spoken reply)

Fast paths:
  - Single conversation task → skip queue, call brain.process_conversational directly.
  - Single task, no dependencies → skip orchestration overhead, run agent directly.
"""

import json
import logging
import re
import threading

from groq import Groq

import config
from agents import build_agent_registry
from models import Task, TaskResult
from task_router import TaskRouter
from task_queue import TaskQueue
from synthesizer import Synthesizer
from planner import NeuralPlanner
from routing_memory import RoutingMemory

try:
    from guardrails import Guardrails
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False

try:
    from metacognition import MetacognitiveScorer
    _METACOG_AVAILABLE = True
except ImportError:
    _METACOG_AVAILABLE = False

try:
    from task_queue import get_journal
    _JOURNAL_AVAILABLE = True
except ImportError:
    _JOURNAL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Decomposition prompt ───────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a task decomposition engine for a voice assistant called KOBRA.
Break the user's command into the MINIMUM number of subtasks needed.

Agents available:
  - conversation  → answer questions, opinions, jokes, general knowledge (no actions)
  - system        → open apps, run shell commands, system info, folders, keyboard/mouse,
                    window management, focus modes (coding/gaming/research/break)
  - web           → search the web, open URLs, weather, news
  - dev           → simple file creation, open VS Code, run git/pytest/pip commands,
                    explain visible code (VS Code mode), fix terminal errors, commit
  - interpreter   → complex programming tasks: scaffold full projects, write scripts,
                    automate workflows, generate and execute code dynamically.
                    Use for: "build me a X", "write a script that", "automate Y",
                    "create a project called Z", "make a function that", "generate code"
  - media         → play music, stream audio, YouTube, control playback (local/browser)
  - memory        → save or recall facts from long-term memory
  - integration   → send/read Gmail, create/check Google Calendar events,
                    control Spotify via API, save/look up contacts
  - browser       → send WhatsApp messages, read WhatsApp, automate web UI tasks
  - research      → deep web research on a topic: searches, scrapes, synthesizes a full
                    markdown report. Use for: "research X", "do a deep dive on X",
                    "investigate X", "give me a full breakdown of X"
  - screen        → vision-based screen navigation: read screen, click UI elements,
                    fill forms. Use for: "what's on screen", "click the X button",
                    "read the error on screen"
  - knowledge     → search personal knowledge base (indexed local files, projects, notes).
                    Use for: "what did I write about X", "find my notes on X",
                    "what's in my documents about X"
  - mcp           → external services via MCP: GitHub, Notion, Supabase, etc.
                    Use for: "create a GitHub issue", "add to Notion", "query Supabase"

CRITICAL DECOMPOSITION RULES:
- Most commands are a SINGLE task. Default to one task unless the command explicitly
  requires multiple independent actions (e.g. "play music AND open Chrome").
- NEVER split a single intent into multiple tasks. "Play Eminem" = 1 task (media).
  "Help me plan a workout" = 1 task (conversation). "What's the weather" = 1 task (web).
- Only create multiple tasks when the user literally asks for multiple distinct things.

SEARCH + ACTION WORKFLOWS (CRITICAL):
When the user asks to search/find/look up information AND then send/email/save/create something:
- ALWAYS create TWO tasks with a DEPENDENCY
- Task 1: Search for the information (web agent) → can_parallelize: false
- Task 2: Send/email/save the results (integration/dev agent) → can_parallelize: false, depends_on: ["t1"]
- Task 2's instruction must explicitly reference receiving the search results from Task 1

Examples:
  "search top songs and email to john"
    → t1: web agent "Search web for top trending songs of 2026 and return formatted list"
    → t2: integration agent "Send email to john with subject 'Top Songs 2026' containing the song list from the search results" depends_on: ["t1"]

  "find weather and send to mom"
    → t1: web agent "Get current weather forecast"
    → t2: integration agent "Send email to mom with the weather information" depends_on: ["t1"]

- "can_parallelize": true means this task does NOT depend on another task's output.
- "depends_on": list of task IDs that must finish before this task starts.
- Keep each "instruction" self-contained — the agent only sees its own instruction.

WEBSITE / APP / PROJECT RULE — HIGHEST PRIORITY:
- ANY request to build a website, web app, landing page, portfolio, dashboard, or multi-file
  project MUST go to the interpreter agent — NEVER to dev agent.
- Examples that MUST route to interpreter:
    "Create a website for a coffee shop" → interpreter
    "Build me a landing page for my startup" → interpreter
    "Make a portfolio website" → interpreter
    "Create a React app" → interpreter
    "Build a Python Flask app" → interpreter
- Instruction to interpreter should include ALL context: business name, type, style,
  any pages mentioned, and: "Build a complete, professional, multi-file project."

FILE CREATION RULE:
- "Create a file called X and fill it with Y" = ONE task to dev agent.
  Write the instruction as: "Create a file called X at the desktop path and fill it with [content description].
  Generate the content yourself using your knowledge — do not search separately."
- NEVER split create+fill into separate tasks. The dev agent generates the content itself.
- Exception: only split if the user explicitly says "search the web first, then save".

FOLLOW-UP RULE:
- If the command is a follow-up (uses "that", "it", "the first one", "tell me more",
  "which one", "explain", etc.), resolve the reference using the recent conversation context
  and write a fully explicit instruction.

SEARCH-THEN-ACT CHAINING RULE (CRITICAL):
- When the user wants to FIRST fetch/search data and THEN do something with the result
  (email it, save it to a file, send it, post it), create 2 tasks with depends_on.
  Examples:
    "find top trending songs and email them to x@y.com"
      → t1: web agent — "Search the web for top trending songs of 2026. Return a clean numbered list."
             can_parallelize: true, depends_on: []
      → t2: integration agent — "Send an email to x@y.com. The email subject should be 'Top Trending Songs 2026'. The body should contain the list of songs provided in the context from the previous step."
             can_parallelize: false, depends_on: ["t1"]
    "look up [X] and save it to a file" → t1: web, t2: dev (depends_on t1)
    "research [X] and write a summary" → t1: research/web, t2: dev (depends_on t1)
- ONLY set depends_on when t2 genuinely NEEDS t1's output data.
- For independent parallel actions ("play music AND open Chrome"), use can_parallelize: true on both with no depends_on.

CALENDAR RULE:
- "Do I have meetings today or tomorrow?" = ONE task to integration agent.
  NEVER split a time-range query into separate tasks per day.

- Return ONLY valid JSON — no explanation, no markdown fences, nothing else.

JSON schema:
{
  "tasks": [
    {
      "id": "t1",
      "agent": "web",
      "instruction": "Search the web for what LangGraph is and return a concise summary.",
      "can_parallelize": true,
      "depends_on": []
    }
  ]
}
"""


class Orchestrator:
    def __init__(
        self,
        brain,
        memory,
        tool_registry: dict,
        credential_store=None,
        contact_store=None,
        retriever=None,
        mcp_client=None,
    ) -> None:
        self._brain = brain
        self._memory = memory
        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._agents = build_agent_registry(
            brain, memory,
            credential_store=credential_store,
            contact_store=contact_store,
            retriever=retriever,
            mcp_client=mcp_client,
        )
        self._router = TaskRouter()
        self._queue = TaskQueue()
        self._synthesizer = Synthesizer(brain)
        self._planner = NeuralPlanner()
        self._routing_memory = RoutingMemory()
        self._guardrails = Guardrails() if _GUARDRAILS_AVAILABLE else None
        self._metacog = MetacognitiveScorer() if _METACOG_AVAILABLE else None
        # Wire routing memory into brain's memory router for few-shot context
        try:
            self._brain.set_routing_memory(self._routing_memory)
        except Exception:
            pass
        logger.info("Neural Planner + Routing Memory + Guardrails + Metacognition initialized.")
        logger.info("Orchestrator ready — %d agents registered: %s",
                    len(self._agents), sorted(self._agents.keys()))

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, transcript: str, abort_flag: threading.Event) -> str:
        """
        Full pipeline: transcript → spoken response string.
        Called from main.py for every active-state user command.
        """
        logger.info("[ORCHESTRATOR] Input: %r", transcript[:100])

        # Step 0: Build recent context (shared by planner + decomposer)
        recent_ctx = self._build_recent_context()

        # Step 1: Neural Planner — enrich/clarify complex commands
        enriched = self._planner.enrich(transcript, recent_ctx)
        if enriched != transcript:
            logger.info("[ORCHESTRATOR] Planner enriched transcript: %r", enriched[:100])

        # Step 2: Decompose — fast path first, then LLM (with routing memory few-shots)
        raw_tasks = self._fast_path(enriched) or self._decompose(enriched, recent_ctx)

        # Step 3: Validate / enrich
        tasks = self._router.validate_and_enrich(raw_tasks, enriched)
        logger.info("[ORCHESTRATOR] %d task(s): %s",
                    len(tasks), [f"{t.id}→{t.agent_name}" for t in tasks])

        # Step 4: Guardrail check — block destructive/injected instructions
        if self._guardrails:
            for task in tasks:
                gr = self._guardrails.check_instruction(task.instruction)
                if not gr.allowed:
                    if _JOURNAL_AVAILABLE:
                        get_journal().log_guardrail_block(gr.rule, transcript, gr.reason)
                    logger.warning("[ORCHESTRATOR] Guardrail blocked task %s: %s", task.id, gr.reason)
                    return self._guardrails.describe_block(gr)

        # Step 4b: Confirmation guard — irreversible actions need explicit user sign-off
        # If the user's transcript IS itself a confirmation ("yes", "go ahead", "confirm",
        # "do it", "send it") and there are pending confirmations, approve and re-run.
        if self._guardrails:
            _CONFIRM_WORDS = {"yes", "yeah", "yep", "go ahead", "confirm", "do it",
                              "send it", "sure", "affirmative", "proceed", "ok", "okay"}
            _is_confirm = transcript.lower().strip().rstrip(".,!") in _CONFIRM_WORDS or \
                          any(w in transcript.lower() for w in ("yes do it", "yes send", "go ahead", "confirmed"))

            for task in tasks:
                question = self._guardrails.needs_confirmation(task.agent_name, task.instruction)
                if question:
                    if _is_confirm:
                        # User already said yes — mark confirmed and proceed
                        self._guardrails.confirm_pending(task.agent_name, task.instruction)
                        logger.info("[ORCHESTRATOR] Confirmation received for task %s.", task.id)
                    else:
                        # Ask the user first — store pending context so next turn can resolve
                        logger.info("[ORCHESTRATOR] Confirmation required for task %s: %s",
                                    task.id, question)
                        return question

        # Step 5: Metacognitive pre-flight — check confidence, ask clarification if needed
        if self._metacog and len(tasks) > 0:
            scores = self._metacog.score_tasks(tasks, recent_ctx)
            self._metacog.summary_log(scores)
            clarification = self._metacog.get_clarification_prompt(scores)
            if clarification:
                logger.info("[ORCHESTRATOR] Metacog: clarification needed.")
                if _JOURNAL_AVAILABLE:
                    get_journal().log_metacognition(
                        tasks[0].id,
                        min(s.confidence for s in scores),
                        [s.clarification_question for s in scores if s.clarification_needed],
                    )
                return clarification

        # Step 6: Journal the command
        if _JOURNAL_AVAILABLE:
            get_journal().log_command(transcript, enriched, len(tasks))

        # Fast path A: single conversational task
        if len(tasks) == 1 and tasks[0].agent_name == "conversation":
            if abort_flag.is_set():
                return "Understood, sir."
            result = self._brain.process_conversational(tasks[0].instruction)
            self._routing_memory.log_routing(transcript, "conversation", tasks[0].instruction)
            # Store episode in memory router
            try:
                self._brain._memory_router.store_episode_from_result(
                    transcript, "conversation", result, True
                )
            except Exception:
                pass
            return result

        # Fast path B: single non-conversational task (skip queue overhead)
        if len(tasks) == 1:
            agent = self._agents.get(tasks[0].agent_name)
            if agent:
                result = agent.execute(tasks[0], abort_flag)
                outcome = "success" if result.success else "failure"
                self._routing_memory.log_routing(
                    transcript, tasks[0].agent_name, tasks[0].instruction, outcome
                )
                try:
                    self._brain._memory_router.store_episode_from_result(
                        transcript, tasks[0].agent_name, result.output or "", result.success
                    )
                except Exception:
                    pass
                return self._synthesizer.synthesize(transcript, [result])

        # Standard path: multi-task orchestration
        results = self._queue.execute(tasks, self._agents, abort_flag)
        for result in results:
            outcome = "success" if result.success else "failure"
            self._routing_memory.log_routing(
                transcript, result.agent_name, "", outcome
            )
            try:
                self._brain._memory_router.store_episode_from_result(
                    transcript, result.agent_name, result.output or "", result.success
                )
            except Exception:
                pass
        return self._synthesizer.synthesize(transcript, results)

    # ── Decomposition ──────────────────────────────────────────────────────────

    def _build_recent_context(self, limit: int = 4) -> str:
        """
        Build a brief conversation context string for the decomposition model.
        This lets it resolve pronouns and follow-up references like
        'tell me more about that' or 'what was the second story?'.
        """
        turns = self._memory.get_recent(limit=limit)
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "User" if t["role"] == "user" else "KOBRA"
            # Truncate long entries so we don't bloat the decomposition prompt
            lines.append(f"{role}: {t['content'][:300]}")
        return "Recent conversation (for resolving follow-ups):\n" + "\n".join(lines) + "\n\n"

    def _decompose(self, transcript: str, recent_ctx: str = "") -> list[Task] | None:
        """
        Ask Groq (fast model) to break the transcript into subtasks.
        Injects recent conversation history and routing memory few-shots.
        Returns None on any failure — TaskRouter will use rule-based fallback.
        """
        if not recent_ctx:
            recent_ctx = self._build_recent_context()

        # Get few-shot routing examples from memory
        few_shots = self._routing_memory.get_few_shot_examples(transcript, limit=3)

        user_content = ""
        if few_shots:
            user_content += few_shots
        if recent_ctx:
            user_content += recent_ctx
        user_content += f"Current command: {transcript}"

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.1,      # low temp for deterministic structured output
                timeout=30,
            )
            raw = (response.choices[0].message.content or "").strip()
            return self._parse_tasks(raw)

        except Exception as exc:
            logger.warning("[ORCHESTRATOR] Decomposition call failed: %s", exc)
            return None

    def _parse_tasks(self, raw: str) -> list[Task] | None:
        """Parse the Groq JSON response into a list of Task objects."""
        try:
            # Strip markdown fences if the model wraps output in ```json ... ```
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            task_list = data.get("tasks", [])
            if not task_list:
                return None

            tasks: list[Task] = []
            for item in task_list:
                tasks.append(Task(
                    id=str(item.get("id", f"t{len(tasks)+1}")),
                    agent_name=str(item.get("agent", "conversation")),
                    instruction=str(item.get("instruction", "")),
                    can_parallelize=bool(item.get("can_parallelize", True)),
                    depends_on=[str(d) for d in item.get("depends_on", [])],
                ))
            return tasks if tasks else None

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("[ORCHESTRATOR] JSON parse failed: %s | raw: %.200s", exc, raw)
            return None

    # ── Pre-decomposition fast paths ───────────────────────────────────────────

    def _fast_path(self, transcript: str) -> list[Task] | None:
        """
        Pattern-match common command shapes that the LLM tends to over-split,
        and return a single Task directly — skipping the Groq decomposition call.

        Currently handles:
          - "create/make/write a file … and fill/put/add content"
            → single dev task so DevAgent generates content in one create_file call
        """
        t = transcript.lower()

        # ── Bail out early for complex project creation — let interpreter handle it ──
        _PROJECT_SIGNALS = (
            "website", "web app", "web application", "landing page", "webpage",
            "web page", "project", "app for", "application for", "dashboard",
            "portfolio", "full stack", "frontend", "backend",
        )
        if any(w in t for w in _PROJECT_SIGNALS):
            logger.info("[ORCHESTRATOR] fast_path: project/website → interpreter (skip)")
            return None

        # ── Create-and-fill: single file with content ──────────────────────────
        has_create = any(w in t for w in (
            "create", "make", "write", "generate", "save", "new file",
        ))
        has_file   = any(w in t for w in (
            "file", ".txt", ".md", ".py", ".json", ".csv", ".html",
            ".ts", ".js", ".yaml", ".yml", ".toml",
        ))
        has_fill   = any(w in t for w in (
            "fill", "put", "add", "include", "with", "about",
            "containing", "content", "information", "info", "details",
            "latest", "all the", "everything",
        ))

        if has_create and has_file and has_fill:
            logger.info("[ORCHESTRATOR] fast_path: create+fill → single dev task")
            return [Task(
                id="t1",
                agent_name="dev",
                instruction=transcript,
                can_parallelize=False,
                depends_on=[],
            )]

        # ── Search + Email workflow: search then send email with results ─────────
        _SEARCH_PATTERNS = ("search ", "look up ", "find ", "get ", "list ", "what are ", "top ", "trending ")
        _EMAIL_PATTERNS = ("send email", "email to", "send mail", "mail to", "email ")

        has_search = any(w in t for w in _SEARCH_PATTERNS)
        has_email = any(w in t for w in _EMAIL_PATTERNS)
        connector_words = (" and ", " then ", " after that ")
        has_connector = any(w in t for w in connector_words)

        if has_search and has_email and has_connector:
            logger.info("[ORCHESTRATOR] fast_path: search+email → two tasks with dependency")
            # Extract email recipient
            email_match = re.search(r'(?:email|mail)\s+(?:to\s+)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', t)
            recipient = email_match.group(1) if email_match else "the user"

            # Create two tasks with dependency
            return [
                Task(
                    id="t1",
                    agent_name="web",
                    instruction=f"Search the web for: '{transcript}'. Extract and format the key information as a clear list.",
                    can_parallelize=False,
                    depends_on=[],
                ),
                Task(
                    id="t2",
                    agent_name="integration",
                    instruction=f"Send an email to {recipient} with subject 'Search Results' and body containing the search results from the previous task. Use the exact results provided in the context.",
                    can_parallelize=False,
                    depends_on=["t1"],
                ),
            ]

        # ── Search + Save/Create file workflow ──────────────────────────────────
        _SAVE_PATTERNS = ("save ", "create file", "write to file", "make file", "put in file")
        has_save = any(w in t for w in _SAVE_PATTERNS)

        if has_search and has_save and has_connector:
            logger.info("[ORCHESTRATOR] fast_path: search+save → two tasks with dependency")
            return [
                Task(
                    id="t1",
                    agent_name="web",
                    instruction=f"Search the web for: '{transcript}'. Extract and format the key information.",
                    can_parallelize=False,
                    depends_on=[],
                ),
                Task(
                    id="t2",
                    agent_name="dev",
                    instruction="Create a file with the search results from the previous task. Save it to the desktop with an appropriate filename.",
                    can_parallelize=False,
                    depends_on=["t1"],
                ),
            ]

        return None
