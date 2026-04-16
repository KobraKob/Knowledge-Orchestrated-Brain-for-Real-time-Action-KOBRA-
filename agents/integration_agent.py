"""
agents/integration_agent.py — API-based external service agent for KOBRA.

Handles:
  - Gmail: send_email, read_emails
  - Google Calendar: create_calendar_event, get_calendar_events, delete_calendar_event
  - Spotify: play_spotify, control_spotify
  - Contacts: save_contact, resolve_contact

All integration modules are lazily authenticated — first action triggers OAuth if needed.
Contact resolution errors prompt KOBRA to ask the user for the missing info.
"""

import logging

import config
from agents.base_agent import BaseAgent
from contact_store import ContactStore, ContactNotFoundError
from credential_store import CredentialStore
from models import Task

try:
    from integrations.base_integration import NotAuthenticatedError, IntegrationError
    _BASE_INTEGRATION_AVAILABLE = True
except ImportError:
    _BASE_INTEGRATION_AVAILABLE = False
    class NotAuthenticatedError(Exception):
        def __init__(self, *args, **kwargs):
            self.service = kwargs.get("service", "unknown")
            super().__init__(*args)
    class IntegrationError(Exception):
        pass

try:
    from integrations.gmail import GmailIntegration
    _GMAIL_AVAILABLE = True
except ImportError:
    _GMAIL_AVAILABLE = False
    GmailIntegration = None

try:
    from integrations.google_calendar import GoogleCalendarIntegration
    _CALENDAR_AVAILABLE = True
except ImportError:
    _CALENDAR_AVAILABLE = False
    GoogleCalendarIntegration = None

try:
    from integrations.spotify import SpotifyIntegration, NoActiveDeviceError
    _SPOTIFY_AVAILABLE = True
except ImportError:
    _SPOTIFY_AVAILABLE = False
    SpotifyIntegration = None
    class NoActiveDeviceError(Exception):
        pass

logger = logging.getLogger(__name__)


class IntegrationAgent(BaseAgent):
    AGENT_NAME = "integration"
    OWNED_TOOLS = [
        "send_email",
        "read_emails",
        "create_calendar_event",
        "get_calendar_events",
        "delete_calendar_event",
        "play_spotify",
        "control_spotify",
        "save_contact",
        "resolve_contact",
        "speak_only",
    ]

    SYSTEM_PROMPT = """\
You are KOBRA's integration agent. You connect to external services: Gmail, Google Calendar, and Spotify.

AVAILABLE TOOLS — use ONLY these exact names, character-for-character:
  send_email(to_name, subject, body)
  read_emails(count, query)
  create_calendar_event(title, date, time, duration_minutes, attendees, description)
  get_calendar_events(date, count)
  delete_calendar_event(event_title, date)
  play_spotify(query)
  control_spotify(action, volume)
  save_contact(name, aliases, email, phone, whatsapp)
  resolve_contact(name)
  speak_only(response)

RULES:
- NEVER call a tool not in the list above. NEVER add spaces, prefixes ("check_", "do_", "fetch_"),
  or suffixes to tool names. Copy the name exactly as written.
- For email tasks: use send_email or read_emails.
- For calendar tasks: use get_calendar_events (NOT check_calendar, NOT fetch_events — exactly: get_calendar_events).
- For Spotify: use play_spotify or control_spotify.
- To look up a contact: use resolve_contact. To save one: use save_contact.
- NEVER fabricate contact info not provided in the instruction.
- All output spoken aloud — no markdown, no bullets, no URLs. Max 2 sentences.
"""

    def __init__(
        self,
        brain,
        memory,
        credential_store: CredentialStore,
        contact_store: ContactStore,
    ) -> None:
        super().__init__(brain, memory)
        self._creds = credential_store
        self._contacts = contact_store
        self._gmail = GmailIntegration(credential_store, contact_store) if _GMAIL_AVAILABLE else None
        self._calendar = GoogleCalendarIntegration(credential_store, contact_store) if _CALENDAR_AVAILABLE else None
        self._spotify = SpotifyIntegration(credential_store) if _SPOTIFY_AVAILABLE else None

    # ── Tool trimming ──────────────────────────────────────────────────────────

    def _select_tools_for(self, instruction: str) -> list[str]:
        """
        Return the minimal set of tools needed for this instruction (≤4 tools).
        Showing all 10 integration tools to the 70b model is fine, but keeping
        the list tight makes tool selection faster and eliminates wrong-tool picks.
        """
        t = instruction.lower()
        base = ["speak_only"]

        if any(w in t for w in ("email", "gmail", "send mail", "inbox", "unread")):
            return base + ["send_email", "read_emails", "save_contact"]

        if any(w in t for w in ("calendar", "meeting", "event", "schedule",
                                 "appointment", "today", "tomorrow", "this week")):
            return base + ["get_calendar_events", "create_calendar_event", "delete_calendar_event"]

        if any(w in t for w in ("spotify", "play", "pause", "skip", "volume",
                                 "resume", "music", "song", "track")):
            return base + ["play_spotify", "control_spotify"]

        if any(w in t for w in ("contact", "save", "who is", "number", "phone")):
            return base + ["save_contact", "resolve_contact"]

        # Fallback — most common pair
        return base + ["get_calendar_events", "send_email", "play_spotify"]

    # ── Task execution ─────────────────────────────────────────────────────────

    def _run(self, task: Task) -> str:
        instruction = self._build_instruction(task)

        # Pick only the tools relevant to this specific instruction.
        # Showing 10 tools → model hallucinates names. Showing 3-4 → it picks correctly.
        scoped_tool_names = self._select_tools_for(instruction)

        # Register integration tool callables so brain._dispatch_tool() can call them
        extra_tools = self._make_tool_callables()
        original_registry = dict(self._brain._registry)
        self._brain._registry.update(extra_tools)

        try:
            result = self._brain.process_scoped(
                instruction,
                scoped_tool_names,
                self.SYSTEM_PROMPT,
                model=config.GROQ_MODEL_TOOLS,   # 70b — integration needs reliable tool selection
            )
        except NotAuthenticatedError as exc:
            result = (
                f"I need to connect to {exc.service} first, sir. "
                f"I'll open the login page now."
            )
            # Trigger auth flow (opens browser)
            if exc.service == "gmail":
                self._gmail.ensure_authenticated()
            elif exc.service == "google_calendar":
                self._calendar.ensure_authenticated()
            elif exc.service == "spotify":
                self._spotify.ensure_authenticated()
        except ContactNotFoundError as exc:
            result = (
                f"I don't have contact info for {exc.name!r}, sir. "
                f"Tell me their email and I'll save it for future use."
            )
        except NoActiveDeviceError as exc:
            result = str(exc)
        except IntegrationError as exc:
            result = f"Integration error, sir: {exc}"
        finally:
            # Restore original registry
            self._brain._registry.clear()
            self._brain._registry.update(original_registry)

        return result

    # ── Tool callables (injected into brain registry during execution) ──────────

    def _make_tool_callables(self) -> dict:
        gmail = self._gmail
        calendar = self._calendar
        spotify = self._spotify
        contacts = self._contacts

        def send_email(to_name: str, subject: str, body: str) -> str:
            if not _GMAIL_AVAILABLE or gmail is None:
                return "Gmail integration not available — integrations module not installed."
            return gmail.send_email(to_name, subject, body)

        def read_emails(count: int = 5, query: str = "") -> str:
            if not _GMAIL_AVAILABLE or gmail is None:
                return "Gmail integration not available — integrations module not installed."
            return gmail.read_emails(count=count, query=query)

        def create_calendar_event(
            title: str, date: str, time: str,
            duration_minutes: int = 60,
            attendees: list | None = None,
            description: str = "",
        ) -> str:
            if not _CALENDAR_AVAILABLE or calendar is None:
                return "Google Calendar integration not available — integrations module not installed."
            return calendar.create_event(
                title, date, time,
                duration_minutes=duration_minutes,
                attendees=attendees or [],
                description=description,
            )

        def get_calendar_events(date: str = "today", count: int = 5) -> str:
            if not _CALENDAR_AVAILABLE or calendar is None:
                return "Google Calendar integration not available — integrations module not installed."
            return calendar.get_events(date=date, count=count)

        def delete_calendar_event(event_title: str, date: str = "today") -> str:
            if not _CALENDAR_AVAILABLE or calendar is None:
                return "Google Calendar integration not available — integrations module not installed."
            return calendar.delete_event(event_title, date=date)

        def play_spotify(query: str) -> str:
            if not _SPOTIFY_AVAILABLE or spotify is None:
                return "Spotify integration not available — integrations module not installed."
            return spotify.play(query)

        def control_spotify(action: str, volume: int | None = None) -> str:
            if not _SPOTIFY_AVAILABLE or spotify is None:
                return "Spotify integration not available — integrations module not installed."
            action = action.lower()
            if action == "pause":
                return spotify.pause()
            elif action == "resume":
                return spotify.resume()
            elif action == "skip":
                return spotify.skip()
            elif action == "previous":
                return spotify.previous()
            elif action == "current_track":
                return spotify.get_current_track()
            elif action == "set_volume" and volume is not None:
                return spotify.set_volume(volume)
            else:
                return f"Unknown Spotify action: {action}"

        def save_contact(
            name: str,
            aliases: list | None = None,
            email: str | None = None,
            phone: str | None = None,
            whatsapp: str | None = None,
        ) -> str:
            contacts.save_contact(
                name,
                aliases=aliases or [],
                email=email,
                phone=phone,
                whatsapp=whatsapp,
            )
            return f"Contact saved: {name}."

        def resolve_contact(name: str) -> str:
            contact = contacts.resolve(name)
            if contact is None:
                return f"No contact found for {name!r}."
            return contacts.format_for_voice(contact)

        return {
            "send_email":             send_email,
            "read_emails":            read_emails,
            "create_calendar_event":  create_calendar_event,
            "get_calendar_events":    get_calendar_events,
            "delete_calendar_event":  delete_calendar_event,
            "play_spotify":           play_spotify,
            "control_spotify":        control_spotify,
            "save_contact":           save_contact,
            "resolve_contact":        resolve_contact,
        }
