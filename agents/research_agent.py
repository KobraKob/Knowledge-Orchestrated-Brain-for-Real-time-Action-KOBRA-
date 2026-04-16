"""
agents/research_agent.py — Deep web research agent for KOBRA.

Flow (original pipeline — used for standard deep research tasks):
  1. Decompose topic into 5 targeted search queries (Groq)
  2. Run all queries in parallel (ThreadPoolExecutor)
  3. Scrape top URLs from each result set (Playwright — single shared session)
  4. Synthesize all scraped content into a structured markdown report (Groq)
  5. Save report to Desktop, open in VS Code, speak a summary

ReAct Loop (v4):
  For open-ended research tasks, the ReAct loop provides iterative
  search → read → synthesize cycles with a single shared Playwright browser
  (fixes the memory exhaustion bug from opening a new browser per URL).

Trigger phrases (handled by task_router.py):
  "research ...", "do a deep dive on ...", "investigate ...",
  "give me a full breakdown of ...", "analyze ..."
"""

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from agents.base_agent import BaseAgent
from groq import Groq
from models import Task, TaskResult

logger = logging.getLogger(__name__)

_MAX_REACT_STEPS = 5

_REACT_SYSTEM = """\
You are KOBRA's research agent. You search the web and scrape pages to gather information.

At each step, respond with ONLY a JSON object (no markdown fences, no extra text):
{
  "thought": "What I know so far and what I should do next",
  "action": "tool_name",
  "action_input": "input to the tool",
  "final_answer": ""
}

When you have gathered enough information to answer the task, set action to "done" and fill final_answer with a concise, spoken-friendly summary (2-4 sentences, address user as sir).

Available tools:
- web_search: Search the web for information. action_input: the search query string
- scrape_page: Scrape text content from a URL. action_input: the full URL to scrape
- done: Research complete. Fill final_answer with a spoken summary of findings.

Rules:
- Always think before acting — plan what to search and why
- Search multiple angles before concluding
- scrape_page the most relevant URLs from search results to get details
- If a tool fails, try a different query or URL
- Never repeat the exact same action twice
- final_answer must be spoken aloud — no markdown, no bullet points, no URLs
- Be concise: synthesize findings into 2-4 clear sentences
"""

_SYSTEM_PROMPT = """\
You are KOBRA's research module — relentless, thorough, and mercilessly organized.
When given a topic, you search exhaustively, cite sources, and structure findings clearly.
You write reports like a senior analyst who is also slightly impatient with vague questions.
Never invent facts. If something isn't in your retrieved data, say so.
"""


class ResearchAgent(BaseAgent):
    AGENT_NAME = "research"
    OWNED_TOOLS = ["web_search", "open_vscode", "speak_only"]
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    MAX_QUERIES = 5
    MAX_SCRAPE_PER_QUERY = 2
    MAX_CHARS_PER_PAGE = 3000
    MAX_CONTEXT_CHARS = 8000

    def __init__(self, brain, memory) -> None:
        super().__init__(brain, memory)
        self._client = Groq(api_key=config.GROQ_API_KEY)

    # ── ReAct loop ─────────────────────────────────────────────────────────────

    def _react_loop(
        self,
        task_instruction: str,
        abort_flag: threading.Event,
    ) -> str:
        """
        ReAct loop: Observe → Think → Act → Observe → ... → Done

        A single Playwright browser is opened before the loop and shared across
        all scrape_page calls, then closed after — fixes memory exhaustion from
        the original per-URL browser pattern.

        Returns the final answer string (spoken-friendly).
        """
        # Start a shared Playwright browser for this task
        shared_browser = None
        shared_page = None
        playwright_ctx = None

        try:
            from playwright.sync_api import sync_playwright
            playwright_ctx = sync_playwright().start()
            shared_browser = playwright_ctx.chromium.launch(headless=True)
            shared_page = shared_browser.new_page()
            logger.info("[RESEARCH REACT] Shared Playwright browser opened.")
        except Exception as exc:
            logger.warning("[RESEARCH REACT] Could not open Playwright browser: %s. "
                           "scrape_page will be unavailable.", exc)

        tools = self._make_react_tools(shared_page)
        messages = [
            {"role": "system", "content": _REACT_SYSTEM},
            {"role": "user", "content": f"Task: {task_instruction}"},
        ]

        final_result = "I completed the research, sir."

        try:
            for step in range(_MAX_REACT_STEPS):
                if abort_flag.is_set():
                    final_result = "Aborted, sir."
                    break

                try:
                    response = self._client.chat.completions.create(
                        model=config.GROQ_MODEL_TOOLS,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.1,
                        timeout=30,
                    )
                    raw = (response.choices[0].message.content or "").strip()
                    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                    step_data = json.loads(clean)
                except Exception as exc:
                    logger.warning("[RESEARCH REACT] Step %d parse failed: %s", step, exc)
                    break

                thought = step_data.get("thought", "")
                action = step_data.get("action", "done")
                action_input = step_data.get("action_input", "")
                final_answer = step_data.get("final_answer", "")

                logger.info(
                    "[RESEARCH REACT] Step %d — thought: %s | action: %s",
                    step, thought[:60], action,
                )

                # Append assistant reasoning to messages
                messages.append({"role": "assistant", "content": raw})

                if action == "done" or step == _MAX_REACT_STEPS - 1:
                    final_result = final_answer or "Research complete, sir."
                    break

                # Execute the chosen tool
                tool_fn = tools.get(action)
                if not tool_fn:
                    observation = (
                        f"Error: Tool '{action}' not found. "
                        f"Available: {list(tools.keys())}"
                    )
                else:
                    try:
                        result = tool_fn(action_input)
                        if hasattr(result, "output"):
                            observation = result.output if result.success else f"Tool failed: {result.output}"
                        else:
                            observation = str(result)
                    except Exception as exc:
                        observation = f"Tool error: {exc}"

                logger.info("[RESEARCH REACT] Step %d observation: %.100s", step, observation)
                messages.append({"role": "user", "content": f"Observation: {observation}"})

        finally:
            # Always close the shared browser
            if shared_browser is not None:
                try:
                    shared_browser.close()
                    logger.info("[RESEARCH REACT] Shared Playwright browser closed.")
                except Exception:
                    pass
            if playwright_ctx is not None:
                try:
                    playwright_ctx.stop()
                except Exception:
                    pass

        return final_result

    def _make_react_tools(self, shared_page) -> dict:
        """Build the tool callables used by the ReAct loop."""

        def web_search(action_input: str) -> str:
            try:
                result = self._brain._dispatch_tool("web_search", {"query": action_input.strip()})
                return str(result)[:2000]
            except Exception as exc:
                return f"Search error: {exc}"

        def scrape_page(action_input: str) -> str:
            url = action_input.strip()
            if shared_page is None:
                return "Playwright unavailable — cannot scrape page."
            try:
                shared_page.goto(url, timeout=12000, wait_until="domcontentloaded")
                content = shared_page.evaluate("""() => {
                    const main = document.querySelector('article, main, [role="main"]');
                    return (main || document.body).innerText;
                }""")
                return (content or "")[:self.MAX_CHARS_PER_PAGE]
            except Exception as exc:
                logger.debug("[RESEARCH REACT] scrape_page failed for %s: %s", url, exc)
                return f"Failed to scrape {url}: {exc}"

        return {
            "web_search": web_search,
            "scrape_page": scrape_page,
        }

    # ── Original deep-research pipeline ────────────────────────────────────────

    def _run(self, task: Task) -> str:
        topic = self._build_instruction(task)

        # 1. Generate search queries
        queries = self._generate_queries(topic)
        logger.info("[RESEARCH] %d queries for: %r", len(queries), topic[:60])

        # 2. Search + scrape in parallel (single shared browser per query worker)
        raw_content = self._search_and_scrape_parallel(queries)

        # 3. Synthesize
        report = self._synthesize(topic, raw_content)

        # 4. Save report
        filename = self._save_report(topic, report)

        # 5. Open in VS Code + speak summary
        try:
            self._brain._dispatch_tool("open_vscode", {"path": filename})
        except Exception:
            pass

        summary = self._extract_summary(report)
        return f"Research complete, sir. Full report saved to your Desktop. {summary}"

    # ── Step 1: Query generation ───────────────────────────────────────────────

    def _generate_queries(self, topic: str) -> list[str]:
        prompt = (
            f'Generate {self.MAX_QUERIES} specific web search queries to research: "{topic}"\n'
            f"Return ONLY a JSON array of strings. No explanation.\n"
            f'Example: ["query one", "query two", "query three"]'
        )
        try:
            raw = self._brain.process_conversational(prompt)
            # Extract JSON array from the response
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return [q for q in queries if isinstance(q, str)][:self.MAX_QUERIES]
        except Exception as exc:
            logger.warning("Query generation failed: %s", exc)
        # Fallback: basic queries
        return [f"{topic} overview", f"{topic} best practices", f"{topic} comparison 2024"]

    # ── Step 2: Parallel search + scrape ──────────────────────────────────────

    def _search_and_scrape_parallel(self, queries: list[str]) -> list[dict]:
        def search_one(query: str) -> dict:
            search_result = self._brain._dispatch_tool("web_search", {"query": query})
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', str(search_result))
            # Filter out media/social/tracking URLs
            urls = [u for u in urls if not any(s in u for s in
                    ("youtube.com/watch", "twitter.com", "facebook.com",
                     "instagram.com", "redirect", "google.com/search"))]
            scraped = []
            for url in urls[:self.MAX_SCRAPE_PER_QUERY]:
                try:
                    content = self._scrape_url_isolated(url)
                    if content and len(content.strip()) > 100:
                        scraped.append({"url": url, "content": content[:self.MAX_CHARS_PER_PAGE]})
                except Exception as exc:
                    logger.debug("Scrape failed for %s: %s", url, exc)
            return {"query": query, "search_summary": str(search_result)[:500], "scraped": scraped}

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(search_one, q): q for q in queries}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning("Search worker failed: %s", exc)
        return results

    def _scrape_url_isolated(self, url: str) -> str:
        """
        Scrape a URL using an isolated Playwright browser instance.
        Used by the parallel pipeline (_search_and_scrape_parallel) where
        each worker thread needs its own browser to avoid cross-thread issues.
        """
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=10000, wait_until="domcontentloaded")
                content = page.evaluate("""() => {
                    const main = document.querySelector('article, main, [role="main"]');
                    return (main || document.body).innerText;
                }""")
                browser.close()
                return (content or "")[:self.MAX_CHARS_PER_PAGE]
        except Exception as exc:
            logger.debug("Playwright scrape failed for %s: %s", url, exc)
            return ""

    # ── Step 3: Synthesis ─────────────────────────────────────────────────────

    def _synthesize(self, topic: str, raw_content: list[dict]) -> str:
        context_parts = []
        for item in raw_content:
            context_parts.append(f"## Query: {item['query']}\n{item['search_summary']}")
            for scraped in item.get("scraped", []):
                context_parts.append(f"### Source: {scraped['url']}\n{scraped['content']}")

        context = "\n\n".join(context_parts)[:self.MAX_CONTEXT_CHARS]

        prompt = (
            f'Write a comprehensive research report on: "{topic}"\n\n'
            f"Structure the report with:\n"
            f"- ## Executive Summary (3 sentences)\n"
            f"- ## Key Findings (bullet points)\n"
            f"- ## Detailed Analysis (prose)\n"
            f"- ## Recommendations\n"
            f"- ## Sources\n\n"
            f"Base the report ONLY on the research data below. Cite URLs where relevant.\n\n"
            f"RESEARCH DATA:\n{context}"
        )
        return self._brain.process_conversational(prompt)

    # ── Step 4: Save report ───────────────────────────────────────────────────

    def _save_report(self, topic: str, content: str) -> str:
        safe_name = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        username = os.environ.get("USERNAME", "user")
        filename = f"C:/Users/{username}/Desktop/research_{safe_name}_{timestamp}.md"

        header = (
            f"# Research: {topic}\n"
            f"*Generated by KOBRA on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n\n"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + content)

        logger.info("[RESEARCH] Report saved: %s", filename)
        return filename

    def _extract_summary(self, report: str) -> str:
        """Pull the executive summary section for the spoken response."""
        match = re.search(r'Executive Summary.*?\n(.*?)(?=\n##|\Z)', report,
                          re.DOTALL | re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            # Trim to ~2 sentences for spoken output
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            return " ".join(sentences[:2])
        # Fallback: first 200 chars
        return report[:200].strip()

    # ── execute() override — passes abort_flag into ReAct loop ────────────────

    def execute(self, task: Task, abort_flag: threading.Event) -> TaskResult:
        """
        Override execute to route between:
          - ReAct loop: quick spoken research (task starts with "quick research",
            "what is", "tell me about", etc.)
          - Full pipeline: deep research with saved report (everything else)
        The abort_flag is forwarded to the ReAct loop for cooperative cancellation.
        """
        import time

        if abort_flag.is_set():
            logger.info("[%s] Skipped (aborted): %s", self.AGENT_NAME, task.id)
            return TaskResult(
                task_id=task.id,
                agent_name=self.AGENT_NAME,
                success=False,
                output="Task skipped — abort requested.",
                was_aborted=True,
            )

        start = time.perf_counter()
        logger.info("[%s] Starting task %s: %.80s", self.AGENT_NAME, task.id, task.instruction)

        try:
            instruction = self._build_instruction(task)
            instruction_lower = instruction.lower()

            # Use ReAct for quick spoken research queries
            _react_triggers = (
                "quick research", "what is ", "what are ", "who is ",
                "tell me about", "briefly", "short answer", "summarize",
            )
            use_react = any(t in instruction_lower for t in _react_triggers)

            if use_react:
                logger.info("[RESEARCH] Using ReAct loop for quick research: %.80s", instruction)
                output = self._react_loop(instruction, abort_flag)
            else:
                # Full deep-research pipeline
                output = self._run(task)

            success = True

        except Exception as exc:
            logger.exception("[%s] Task %s raised: %s", self.AGENT_NAME, task.id, exc)
            output = f"Error: {exc}"
            success = False

        duration = time.perf_counter() - start
        logger.info("[%s] Task %s done in %.2fs", self.AGENT_NAME, task.id, duration)

        return TaskResult(
            task_id=task.id,
            agent_name=self.AGENT_NAME,
            success=success,
            output=output,
            duration_seconds=duration,
        )
