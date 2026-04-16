"""
integrations/google_calendar.py — Google Calendar integration for KOBRA.

Shares the same OAuth token as Gmail (same CredentialStore key "gmail").
The gmail integration must run its OAuth flow first — it includes the calendar
scope — so this module simply piggybacks on the saved token.

Actions:
  - create_event(title, date, time, duration_minutes, attendees, description)
  - get_events(date, count)
  - delete_event(event_title, date)
"""

import difflib
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config
from contact_store import ContactStore
from credential_store import CredentialStore
from integrations.base_integration import BaseIntegration, IntegrationError

logger = logging.getLogger(__name__)


def _get_timezone():
    tz_name = getattr(config, "TIMEZONE", "UTC")
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, Exception):
        return timezone.utc


class GoogleCalendarIntegration(BaseIntegration):
    SERVICE_NAME = "google_calendar"

    def __init__(self, credential_store: CredentialStore, contact_store: ContactStore) -> None:
        self._creds_store = credential_store
        self._contacts = contact_store
        self._service = None

    # ── Auth ───────────────────────────────────────────────────────────────────

    def ensure_authenticated(self) -> bool:
        """
        Piggybacks on the Gmail OAuth token (same credentials, same scope grant).
        If Gmail is authenticated, Calendar is automatically authenticated.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
        except ImportError:
            logger.error("[CALENDAR] google-api-python-client not installed.")
            return False

        token_data = self._creds_store.load("gmail")
        if not token_data:
            logger.warning("[CALENDAR] No Gmail token found — run Gmail auth first.")
            return False

        try:
            from integrations.gmail import GMAIL_SCOPES
            creds = Credentials.from_authorized_user_info(token_data, GMAIL_SCOPES)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            self._creds = creds
            self._service = None
            return True
        except Exception as exc:
            logger.error("[CALENDAR] Auth failed: %s", exc)
            return False

    def _get_service(self):
        if self._service is None:
            self._require_auth()
            from googleapiclient.discovery import build
            self._service = build("calendar", "v3", credentials=self._creds)
        return self._service

    # ── Actions ────────────────────────────────────────────────────────────────

    def create_event(
        self,
        title: str,
        date: str,
        time: str,
        duration_minutes: int = 60,
        attendees: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Create a calendar event. Resolves attendee names to emails."""
        self._require_auth()

        tz = _get_timezone()
        start_dt = _parse_datetime(date, time, tz)
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # Resolve attendee names to emails
        attendee_list = []
        resolved_names = []
        for name in (attendees or []):
            contact = self._contacts.resolve(name)
            if contact and contact.get("email"):
                attendee_list.append({"email": contact["email"]})
                resolved_names.append(contact["name"])
            else:
                logger.warning("[CALENDAR] Could not resolve attendee: %r", name)

        event_body = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": getattr(config, "TIMEZONE", "UTC"),
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": getattr(config, "TIMEZONE", "UTC"),
            },
        }
        if attendee_list:
            event_body["attendees"] = attendee_list

        try:
            svc = self._get_service()
            created = svc.events().insert(
                calendarId="primary",
                body=event_body,
                sendUpdates="all" if attendee_list else "none",
            ).execute()

            date_str = start_dt.strftime("%A, %B %d at %-I:%M %p").replace(" 0", " ")
            result = f"Done. '{title}' added to your calendar for {date_str}."
            if resolved_names:
                result += f" Invite sent to {', '.join(resolved_names)}."
            return result

        except Exception as exc:
            logger.error("[CALENDAR] create_event failed: %s", exc)
            raise IntegrationError(f"Calendar event creation failed: {exc}") from exc

    def get_events(self, date: str = "today", count: int = 5) -> str:
        """Fetch upcoming events and return a voice-friendly summary.
        date can be 'today', 'tomorrow', 'today and tomorrow', 'this week', etc.
        """
        self._require_auth()

        # Handle compound date like "today and tomorrow"
        if "and" in date.lower():
            parts = [p.strip() for p in date.lower().split("and")]
            results = []
            for part in parts:
                result = self._get_events_for_day(part, count)
                results.append(result)
            return " ".join(results)

        return self._get_events_for_day(date, count)

    def _get_events_for_day(self, date: str, count: int = 5) -> str:
        """Internal — fetch and format events for a single named day."""
        tz = _get_timezone()
        start, end = _day_bounds(date, tz)

        try:
            svc = self._get_service()
            result = svc.events().list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                maxResults=count,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = result.get("items", [])
            day_label = date.strip().lower()
            if day_label in ("", "today"):
                day_label = "today"

            if not events:
                return f"Nothing scheduled for {day_label}, sir."

            parts = []
            for ev in events:
                summary = ev.get("summary", "Untitled event")
                start_raw = ev["start"].get("dateTime") or ev["start"].get("date", "")
                try:
                    dt = datetime.fromisoformat(start_raw)
                    # Windows-safe strftime (no %-I)
                    hour = dt.strftime("%I").lstrip("0") or "12"
                    ampm = dt.strftime("%p")
                    minute = dt.strftime("%M")
                    time_str = f"{hour}:{minute} {ampm}" if minute != "00" else f"{hour} {ampm}"
                except Exception:
                    time_str = start_raw
                parts.append(f"{summary} at {time_str}")

            count_word = "event" if len(parts) == 1 else "events"
            return f"{len(parts)} {count_word} {day_label}: {', '.join(parts)}."

        except Exception as exc:
            logger.error("[CALENDAR] get_events failed: %s", exc)
            raise IntegrationError(f"Could not fetch events: {exc}") from exc

    def delete_event(self, event_title: str, date: str = "today") -> str:
        """Find and delete an event by approximate title match."""
        self._require_auth()

        tz = _get_timezone()
        start, end = _day_bounds(date, tz)

        try:
            svc = self._get_service()
            result = svc.events().list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                maxResults=20,
                singleEvents=True,
            ).execute()

            events = result.get("items", [])
            titles = [ev.get("summary", "") for ev in events]
            matches = difflib.get_close_matches(event_title, titles, n=1, cutoff=0.5)

            if not matches:
                return f"I couldn't find an event called '{event_title}' on your calendar, sir."

            matched_title = matches[0]
            matched_event = next(ev for ev in events if ev.get("summary") == matched_title)
            svc.events().delete(
                calendarId="primary", eventId=matched_event["id"]
            ).execute()

            return f"Deleted '{matched_title}' from your calendar."

        except Exception as exc:
            logger.error("[CALENDAR] delete_event failed: %s", exc)
            raise IntegrationError(f"Could not delete event: {exc}") from exc


# ── Date / time parsing helpers ───────────────────────────────────────────────

def _parse_datetime(date_str: str, time_str: str, tz) -> datetime:
    """Parse natural language date + time into a timezone-aware datetime."""
    try:
        from dateutil import parser as dateutil_parser
        combined = f"{date_str} {time_str}"
        dt = dateutil_parser.parse(combined, fuzzy=True)
    except Exception:
        dt = datetime.now()

    now = datetime.now()
    # If parsed date is in the past, try adding 1 day
    if dt.date() < now.date():
        dt = dt.replace(year=now.year, month=now.month, day=now.day)

    # Natural language day resolution
    lower = date_str.lower().strip()
    if lower == "today":
        dt = dt.replace(**_today_fields(now))
    elif lower == "tomorrow":
        tomorrow = now + timedelta(days=1)
        dt = dt.replace(**_today_fields(tomorrow))
    elif lower.startswith("next "):
        weekday_name = lower[5:].strip()
        dt = _next_weekday(weekday_name, now, dt)

    if hasattr(tz, 'key'):
        return dt.replace(tzinfo=tz)
    return dt.replace(tzinfo=tz)


def _today_fields(d: datetime) -> dict:
    return {"year": d.year, "month": d.month, "day": d.day}


def _next_weekday(name: str, now: datetime, fallback: datetime) -> datetime:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if name in days:
        target = days.index(name)
        current = now.weekday()
        delta = (target - current + 7) % 7 or 7
        next_day = now + timedelta(days=delta)
        return fallback.replace(year=next_day.year, month=next_day.month, day=next_day.day)
    return fallback


def _day_bounds(date_str: str, tz) -> tuple[datetime, datetime]:
    """Return (start_of_day, end_of_day) for the given date string."""
    now = datetime.now()
    lower = date_str.lower().strip()

    if lower in ("today", ""):
        base = now
    elif lower == "tomorrow":
        base = now + timedelta(days=1)
    elif lower == "this week":
        # Monday to Sunday of current week
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=6)
        s = start.replace(hour=0, minute=0, second=0, microsecond=0)
        e = end.replace(hour=23, minute=59, second=59)
        if hasattr(tz, 'key'):
            return s.replace(tzinfo=tz), e.replace(tzinfo=tz)
        return s.replace(tzinfo=tz), e.replace(tzinfo=tz)
    else:
        try:
            from dateutil import parser as dateutil_parser
            base = dateutil_parser.parse(date_str, fuzzy=True)
        except Exception:
            base = now

    start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end = base.replace(hour=23, minute=59, second=59, microsecond=999999)

    if hasattr(tz, 'key'):
        return start.replace(tzinfo=tz), end.replace(tzinfo=tz)
    return start.replace(tzinfo=tz), end.replace(tzinfo=tz)
