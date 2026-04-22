"""
proactive/scanners.py — Data scanners for KOBRA v5 morning briefing.

Each scanner hits one source and returns structured dicts.
All run in parallel in MorningBriefingEngine.run().
All degrade gracefully when APIs/clients are unavailable.

Scanners:
  scan_calendar(cal_client)       → list[dict] — today + tomorrow events
  scan_email(gmail_client)        → list[dict] — unread emails last 12h
  scan_projects(watched_folders)  → list[dict] — active projects with git info
  scan_last_session(episodic)     → dict | None — what was worked on last time
  scan_github(mcp_client)         → list[dict] — PRs, issues, CI failures
  scan_whatsapp(browser_agent)    → list[dict] — unread direct messages
"""

import logging
import os
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any

import config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(fn, *args, default=None, **kwargs):
    """Run fn(*args, **kwargs), return default on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("[SCANNER] %s failed: %s", fn.__name__, exc)
        return default


def extract_signals(subject: str) -> list[str]:
    """Extract urgency signals from an email subject line."""
    signals = []
    s = subject.lower()
    if any(w in s for w in ("urgent", "asap", "immediately", "critical")):
        signals.append("urgent")
    if s.startswith("re:"):
        signals.append("re:")
    if any(w in s for w in ("deadline", "due", "by tomorrow", "by friday", "by monday")):
        signals.append("deadline")
    if any(w in s for w in ("interview", "offer", "payment", "invoice")):
        signals.append("important")
    return signals


def get_git_info(project_path: str) -> dict:
    """Fast git status for a project directory."""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_path, capture_output=True, text=True, timeout=3,
        ).stdout.strip() or "unknown"
        status_out = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_path, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        changes = len(status_out.split("\n")) if status_out else 0
        return {"branch": branch, "changes": changes}
    except Exception:
        return {"branch": "unknown", "changes": 0}


def get_dir_last_modified(path: str) -> datetime:
    """Return the most recent mtime of any file in a directory (shallow)."""
    latest = datetime.fromtimestamp(os.path.getmtime(path))
    try:
        for entry in os.scandir(path):
            if entry.name.startswith("."):
                continue
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                if mtime > latest:
                    latest = mtime
            except Exception:
                continue
    except Exception:
        pass
    return latest


def estimate_completion(project_path: str) -> dict:
    """Heuristic completion estimate from TODOs + README."""
    open_tasks = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in
                   ("node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build")]
        for fname in files:
            if not fname.endswith((".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx")):
                continue
            try:
                with open(os.path.join(root, fname), encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                todos = re.findall(r"#\s*(TODO|FIXME|WIP|HACK)[:\s](.+)", content)
                open_tasks.extend([t[1].strip()[:60] for t in todos[:3]])
            except Exception:
                pass
        if len(open_tasks) >= 6:
            break

    pct, status = 50, "in_progress"
    readme = os.path.join(project_path, "README.md")
    if os.path.exists(readme):
        try:
            with open(readme, encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()
            if "complete" in text or "finished" in text:
                pct, status = 95, "nearly_complete"
            elif "wip" in text or "work in progress" in text:
                pct, status = 35, "in_progress"
            elif open_tasks:
                pct, status = 55, "in_progress"
        except Exception:
            pass

    return {"pct": pct, "status": status, "open_tasks": open_tasks[:3]}


# ── Scanners ──────────────────────────────────────────────────────────────────

def scan_calendar(cal_client) -> list[dict]:
    """Fetch events for today + tomorrow. Returns [] if client unavailable."""
    if not cal_client:
        return []
    try:
        # Try integration agent-style calendar fetch
        if hasattr(cal_client, "get_events_range"):
            now = datetime.now()
            events = cal_client.get_events_range(start=now, end=now + timedelta(days=2))
        elif hasattr(cal_client, "get_events_today"):
            events = cal_client.get_events_today() or []
        else:
            return []

        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        results = []
        for e in events[:config.BRIEFING_MAX_CALENDAR_EVENTS if hasattr(config, "BRIEFING_MAX_CALENDAR_EVENTS") else 4]:
            try:
                start = e.get("start") or e.get("time", "")
                if isinstance(start, datetime):
                    event_date = start.date()
                    time_str = start.strftime("%I:%M %p")
                elif isinstance(start, str):
                    event_date = today
                    time_str = start
                else:
                    event_date = today
                    time_str = "Unknown time"

                results.append({
                    "title": e.get("summary", e.get("title", "Event")),
                    "time": time_str,
                    "date": event_date,
                    "is_today": event_date == today,
                    "is_tomorrow": event_date == tomorrow,
                    "duration_minutes": e.get("duration", 60),
                    "attendees": ", ".join((e.get("attendees", []) or [])[:3]),
                    "description_preview": str(e.get("description", ""))[:100],
                })
            except Exception:
                continue
        return results
    except Exception as exc:
        logger.debug("[SCANNER] scan_calendar error: %s", exc)
        return []


def scan_email(gmail_client) -> list[dict]:
    """Fetch unread emails from last 12 hours. Returns [] if client unavailable."""
    if not gmail_client:
        return []
    try:
        if hasattr(gmail_client, "get_unread"):
            emails = gmail_client.get_unread(hours=12, max_results=10) or []
        else:
            return []

        results = []
        max_emails = getattr(config, "BRIEFING_MAX_EMAILS", 3)
        for e in emails[:max_emails * 2]:  # fetch more, filter below
            subject = e.get("subject", e.get("Subject", ""))
            sender  = e.get("from_name", e.get("sender", e.get("From", "Unknown")))
            preview = e.get("snippet", e.get("preview", e.get("body", "")))[:150]
            results.append({
                "sender":        sender,
                "sender_email":  e.get("from_email", e.get("sender_email", "")),
                "subject":       subject,
                "preview":       preview,
                "time":          e.get("received_at", e.get("date", "")),
                "subject_signals": extract_signals(subject),
            })
        return results[:max_emails]
    except Exception as exc:
        logger.debug("[SCANNER] scan_email error: %s", exc)
        return []


def scan_projects(watched_folders: list[str] | None = None) -> list[dict]:
    """Scan watched folders for active projects (git repos modified recently)."""
    folders = watched_folders or getattr(config, "WATCHED_FOLDERS", [])
    projects = []

    for folder in folders:
        if not os.path.exists(folder):
            continue
        try:
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if not os.path.isdir(path):
                    continue
                # Must be a git repo to be a "project"
                if not os.path.exists(os.path.join(path, ".git")):
                    continue
                try:
                    last_modified = get_dir_last_modified(path)
                    hours_ago = (datetime.now() - last_modified).total_seconds() / 3600
                    if hours_ago > 168:  # skip if not touched in 7 days
                        continue
                    git_info   = get_git_info(path)
                    completion = estimate_completion(path)
                    projects.append({
                        "name":             name,
                        "path":             path,
                        "last_modified_hours": round(hours_ago, 1),
                        "git_branch":       git_info["branch"],
                        "uncommitted_changes": git_info["changes"],
                        "completion_pct":   completion["pct"],
                        "completion_status": completion["status"],
                        "open_tasks":       completion["open_tasks"],
                    })
                except Exception:
                    continue
        except Exception:
            continue

    max_proj = getattr(config, "BRIEFING_MAX_PROJECTS", 3)
    return sorted(projects, key=lambda x: x["last_modified_hours"])[:max_proj]


def scan_last_session(episodic_memory) -> dict | None:
    """Return summary of last KOBRA session."""
    if not episodic_memory:
        return None
    try:
        last = episodic_memory.get_last_session()
        if not last:
            return None
        ended_str = last.get("ended_at", "")
        hours_ago = last.get("hours_ago", 999)
        if not hours_ago:
            try:
                ended_at = datetime.fromisoformat(ended_str)
                hours_ago = (datetime.now() - ended_at).total_seconds() / 3600
            except Exception:
                hours_ago = 999
        return {
            "hours_ago":    round(hours_ago, 1),
            "project":      last.get("primary_project", "unknown"),
            "summary":      last.get("summary", ""),
            "last_action":  last.get("last_action", ""),
            "status":       last.get("completion_status", "unknown"),
            "tools_used":   last.get("tools_used", []),
            "turn_count":   last.get("turn_count", 0),
        }
    except Exception as exc:
        logger.debug("[SCANNER] scan_last_session error: %s", exc)
        return None


def scan_github(mcp_client) -> list[dict]:
    """Fetch open PRs, failing CI, and assigned issues from GitHub via MCP."""
    if not mcp_client:
        return []
    try:
        if not hasattr(mcp_client, "is_connected") or not mcp_client.is_connected("github"):
            return []
        items = []
        try:
            prs = mcp_client.call_tool("github", "list_pull_requests",
                                       {"state": "open", "review_requested": True}) or []
            for pr in prs[:2]:
                items.append({
                    "type": "pr_review_requested",
                    "description": f"PR #{pr.get('number','?')} needs your review: '{pr.get('title','?')}'",
                })
        except Exception:
            pass
        return items
    except Exception as exc:
        logger.debug("[SCANNER] scan_github error: %s", exc)
        return []


def scan_whatsapp(browser_session=None) -> list[dict]:
    """Check WhatsApp Web for unread direct messages (not group chats)."""
    # Placeholder — requires Playwright session to already be open
    # Returns empty rather than crashing if session not available
    return []
