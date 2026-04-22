"""
agents/media_agent.py — Media playback agent.

Handles: play_media, stop_media, control_media, play_youtube, control_volume.
"""

from agents.base_agent import BaseAgent
from models import Task


class MediaAgent(BaseAgent):
    AGENT_NAME = "media"
    OWNED_TOOLS = [
        "play_media",
        "play_on_spotify",
        "stop_media",
        "control_media",
        "play_youtube",
        "control_volume",
        "speak_only",
    ]
    SYSTEM_PROMPT = (
        "You are KOBRA's media agent. Stream and control audio/video for sir. "
        "Return a short 1-sentence confirmation for text-to-speech.\n\n"
        "AVAILABLE TOOLS — use ONLY these exact names:\n"
        "  play_media(query)              — stream audio DIRECTLY through speakers via yt-dlp\n"
        "  stop_media()                   — stop currently playing audio\n"
        "  control_media(action)          — pause/play/next/previous\n"
        "  play_youtube(query)            — open YouTube in the browser and play the video\n"
        "  play_on_spotify(query)         — open Spotify and search for the track\n"
        "  control_volume(action, steps)  — up/down/mute/unmute\n"
        "  speak_only(response)           — respond with no action\n\n"
        "ROUTING RULES — READ EVERY WORD:\n"
        "- Sir says 'on YouTube' or 'YouTube'  →  ALWAYS use play_youtube. No exceptions.\n"
        "- Sir says 'on Spotify' or 'Spotify'  →  ALWAYS use play_on_spotify. No exceptions.\n"
        "- Sir says 'play [song]' with NO platform mentioned  →  use play_media (streams directly).\n"
        "- NEVER substitute one platform for another. If sir says YouTube, use play_youtube.\n"
        "  If sir says Spotify, use play_on_spotify. The word the user said is the law.\n\n"
        "QUERY RULES:\n"
        "- Always use full descriptive queries: 'Kendrick Lamar Not Like Us' not 'Not Like Us'.\n"
        "- stop_media takes NO arguments — call it as stop_media() with empty params.\n"
        "- NEVER add characters, spaces, or JSON inside tool names."
    )

    def _run(self, task: Task) -> str:
        return self._brain.process_scoped(
            self._build_instruction(task),
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )
