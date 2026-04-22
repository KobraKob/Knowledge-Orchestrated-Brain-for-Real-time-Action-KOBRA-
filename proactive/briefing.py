"""
proactive/briefing.py — Morning Briefing Engine for KOBRA v5.

Two-phase startup:
  Phase 1: KOBRA speaks greeting immediately (< 1 second)
  Phase 2: Background scan completes, synthesized briefing spoken (10-30s later)

The briefing is NOT a list of facts. It's a natural spoken narrative that:
  - Leads with what matters most (not what came first in the data)
  - Connects related items (meeting + unfinished project → mention together)
  - Flags urgency (deadlines, failing CI, unanswered messages from known contacts)
  - Adapts length to user preference (learned from interrupt history)
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, date

from groq import Groq

import config
from proactive.scanners import (
    scan_calendar, scan_email, scan_projects,
    scan_last_session, scan_github, scan_whatsapp,
)

logger = logging.getLogger(__name__)

_SCAN_TIMEOUT = getattr(config, "BRIEFING_SCAN_TIMEOUT", 15)


def _get_time_of_day() -> str:
    hour = datetime.now().hour
    if hour < 12: return "morning"
    if hour < 17: return "afternoon"
    if hour < 21: return "evening"
    return "night"


class MorningBriefingEngine:
    """
    Orchestrates the full startup scan and synthesizes a spoken briefing.

    Usage:
        engine = MorningBriefingEngine(brain, episodic_memory, semantic_memory,
                                        cal_client, gmail_client, mcp_client)
        briefing_text = engine.run()  # blocks until scan + synthesis complete

    Or non-blocking:
        engine.run_async(callback=speaker.speak)  # calls callback when ready
    """

    def __init__(
        self,
        brain,
        episodic_memory=None,
        semantic_memory=None,
        cal_client=None,
        gmail_client=None,
        mcp_client=None,
        watched_folders=None,
    ) -> None:
        self._brain      = brain
        self._episodic   = episodic_memory
        self._semantic   = semantic_memory
        self._calendar   = cal_client
        self._email      = gmail_client
        self._mcp        = mcp_client
        self._folders    = watched_folders or getattr(config, "WATCHED_FOLDERS", [])
        self._groq       = Groq(api_key=config.GROQ_API_KEY)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> str:
        """
        Run the full scan + briefing synthesis. Blocking.
        Returns a natural spoken briefing string.
        """
        logger.info("[BRIEFING] Starting morning scan...")
        results = self._run_scanners()
        relevant = self._filter_relevant(results)

        if not any(relevant.values()):
            return "Nothing pressing on your radar, sir. Clean slate."

        briefing = self._synthesize_briefing(relevant)
        logger.info("[BRIEFING] Briefing ready: %s", briefing[:80])
        return briefing

    def run_async(self, callback, delay: float = 0.0) -> threading.Thread:
        """
        Run scan in background thread. Calls callback(briefing_text) when done.
        Optional delay (seconds) before starting scan.
        """
        def _worker():
            if delay > 0:
                time.sleep(delay)
            try:
                briefing = self.run()
                callback(briefing)
            except Exception as exc:
                logger.error("[BRIEFING] Async run failed: %s", exc)
        t = threading.Thread(target=_worker, daemon=True, name="kobra-briefing")
        t.start()
        return t

    # ── Phase 1: Parallel scan ─────────────────────────────────────────────────

    def _run_scanners(self) -> dict:
        """Run all 6 scanners in parallel. Each has a hard timeout."""
        scanners = {
            "calendar":     lambda: scan_calendar(self._calendar),
            "email":        lambda: scan_email(self._email),
            "projects":     lambda: scan_projects(self._folders),
            "last_session": lambda: scan_last_session(self._episodic),
            "github":       lambda: scan_github(self._mcp),
            "whatsapp":     lambda: scan_whatsapp(),
        }

        results = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_name = {executor.submit(fn): name for name, fn in scanners.items()}
            for future in future_to_name:
                name = future_to_name[future]
                try:
                    results[name] = future.result(timeout=_SCAN_TIMEOUT)
                    logger.debug("[BRIEFING] Scanner '%s' done.", name)
                except Exception as exc:
                    results[name] = None
                    logger.debug("[BRIEFING] Scanner '%s' failed: %s", name, exc)

        return results

    # ── Phase 2: Relevance filtering ──────────────────────────────────────────

    def _filter_relevant(self, results: dict) -> dict:
        """Drop irrelevant noise. Only include what would actually matter to the user."""
        relevant = {}
        today    = datetime.now().date()
        tomorrow = today + timedelta(days=1)

        # Calendar: only today and tomorrow
        if results.get("calendar"):
            today_events = [
                e for e in results["calendar"]
                if e.get("is_today") or e.get("is_tomorrow")
                   or e.get("date") in (today, tomorrow)
            ]
            if today_events:
                relevant["calendar"] = today_events

        # Email: known contacts OR urgency signals
        if results.get("email"):
            known = set()
            if self._semantic and hasattr(self._semantic, "get_known_contacts"):
                known = self._semantic.get_known_contacts()
            important = [
                e for e in results["email"]
                if e.get("sender", "").lower() in known
                or any(sig in ("urgent", "deadline", "important")
                       for sig in e.get("subject_signals", []))
            ]
            # If nothing flagged, still include first email (user should know about unread)
            if not important and results["email"]:
                important = results["email"][:1]
            if important:
                relevant["email"] = important

        # Projects: only ones touched in last 48h, not marked complete
        if results.get("projects"):
            active = [
                p for p in results["projects"]
                if p.get("last_modified_hours", 999) < 48
                and p.get("completion_status") != "nearly_complete"
            ]
            if active:
                relevant["projects"] = active

        # Last session: include if < 16 hours ago
        if results.get("last_session"):
            sess = results["last_session"]
            if sess and sess.get("hours_ago", 999) < 16:
                relevant["last_session"] = sess

        # GitHub: always include if anything returned
        if results.get("github"):
            relevant["github"] = results["github"]

        # WhatsApp: direct messages only (scanner already filters group chats)
        if results.get("whatsapp"):
            relevant["whatsapp"] = results["whatsapp"]

        return relevant

    # ── Phase 3: Synthesis ────────────────────────────────────────────────────

    def _synthesize_briefing(self, relevant: dict) -> str:
        """
        Turn structured scan data into a natural spoken briefing via Groq.
        NOT a notification list — a human assistant narrative.
        """
        data_sections = []
        today = datetime.now().date()

        # Last session — always lead with this if recent
        if relevant.get("last_session"):
            s = relevant["last_session"]
            data_sections.append(
                f"LAST SESSION ({s['hours_ago']}h ago): "
                f"Was working on '{s.get('project','unknown')}'. "
                f"Summary: {s.get('summary','')[:120]}. "
                f"Status: {s.get('status','unknown')}."
            )

        # Active projects
        if relevant.get("projects"):
            for p in relevant["projects"][:3]:
                tasks_str = "; ".join(p.get("open_tasks", [])[:2]) or "none tracked"
                data_sections.append(
                    f"PROJECT '{p['name']}': {p['completion_pct']}% done, "
                    f"modified {p['last_modified_hours']}h ago, "
                    f"{p['uncommitted_changes']} uncommitted changes. "
                    f"Open tasks: {tasks_str}."
                )

        # Calendar events
        if relevant.get("calendar"):
            for e in relevant["calendar"][:4]:
                when = "today" if e.get("is_today") else "tomorrow"
                attendees = f" with {e['attendees']}" if e.get("attendees") else ""
                data_sections.append(
                    f"CALENDAR: {e['title']} at {e['time']} {when}{attendees}."
                )

        # Emails
        if relevant.get("email"):
            for e in relevant["email"][:3]:
                signals = ", ".join(e.get("subject_signals", [])) or "normal"
                data_sections.append(
                    f"EMAIL from {e['sender']}: '{e['subject']}' "
                    f"(priority: {signals}) — {e['preview'][:80]}."
                )

        # GitHub
        if relevant.get("github"):
            for item in relevant["github"][:2]:
                data_sections.append(f"GITHUB: {item['description']}")

        # WhatsApp
        if relevant.get("whatsapp"):
            for msg in relevant["whatsapp"][:2]:
                data_sections.append(
                    f"WHATSAPP: {msg.get('unread_count',1)} unread from {msg.get('contact_name','someone')}."
                )

        if not data_sections:
            return "All clear on your radar, sir."

        data_block = "\n".join(data_sections)

        # Get user preferences
        prefs = {}
        if self._semantic and hasattr(self._semantic, "get_all_preferences"):
            prefs = self._semantic.get_all_preferences()
        prefers_brief = prefs.get("briefing_length", "standard") == "brief"
        lead_with = prefs.get("briefing_lead_with", "")
        time_of_day = _get_time_of_day()

        length_instruction = "2-3 sentences maximum" if prefers_brief else "4-6 sentences, no bullet points"
        lead_instruction   = f"Lead with {lead_with} information." if lead_with else "Lead with what matters most urgency-wise."

        prompt = f"""You are KOBRA, a sharp personal AI assistant with dry wit and deep familiarity with your user.
You've just scanned all of their information. Deliver a morning briefing.

RULES:
- Speak like a real assistant who actually checked this stuff — not a notification list
- {lead_instruction}
- Connect related items: if there's a meeting about a project, mention both together
- Be conversational: not "You have 3 calendar events" but "You've got that design review this afternoon"
- Include one sharp observation or recommendation if something stands out
- Address user as "sir"
- Length: {length_instruction}
- Tone: {time_of_day} energy
- NEVER read raw data — synthesize it into natural speech
- If something was left half-done, say so with appropriate urgency

SCANNED DATA:
{data_block}

Deliver the briefing now:"""

        try:
            response = self._groq.chat.completions.create(
                model=config.GROQ_MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
                temperature=0.85,
                timeout=20,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("[BRIEFING] Groq synthesis failed: %s", exc)
            # Fallback: plain text summary
            return self._plain_fallback(relevant)

    def _plain_fallback(self, relevant: dict) -> str:
        """Fallback if Groq fails — plain text summary."""
        parts = []
        if relevant.get("last_session"):
            s = relevant["last_session"]
            parts.append(f"You were working on {s.get('project','something')} about {s['hours_ago']}h ago.")
        if relevant.get("calendar"):
            e = relevant["calendar"][0]
            parts.append(f"You have {e['title']} at {e['time']}.")
        if relevant.get("email"):
            parts.append(f"You have {len(relevant['email'])} important email(s) unread.")
        return " ".join(parts) or "Morning, sir. Nothing flagged."


# ── Post-briefing feedback detection ──────────────────────────────────────────

def handle_post_briefing_response(transcript: str, semantic_memory) -> None:
    """
    Detect user response to briefing and update preferences.
    Call this from main.py after briefing is spoken and user responds.
    """
    if not semantic_memory or not hasattr(semantic_memory, "update_preference"):
        return
    t = transcript.lower()
    if any(w in t for w in ("what else", "anything else", "tell me more", "more")):
        semantic_memory.update_preference("briefing_length", "detailed")
    elif any(w in t for w in ("got it", "thanks", "okay", "ok", "perfect", "good")):
        semantic_memory.update_preference("briefing_length", "brief")
    elif any(w in t for w in ("too long", "shorter", "brief", "quick")):
        semantic_memory.update_preference("briefing_length", "brief")
