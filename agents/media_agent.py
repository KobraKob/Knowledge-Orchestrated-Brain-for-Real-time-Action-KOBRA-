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
        "  play_youtube(query)            — open YouTube in browser (for WATCHING video only)\n"
        "  control_volume(action, steps)  — up/down/mute/unmute\n"
        "  speak_only(response)           — respond with no action\n\n"
        "CRITICAL RULES:\n"
        "- Use play_media for ALL music/audio requests unless sir says 'on Spotify'.\n"
        "- Use play_on_spotify when sir says 'on Spotify', 'in Spotify', or 'open Spotify and play'.\n"
        "- NEVER use play_youtube for music. play_youtube is ONLY for 'watch a video' requests.\n"
        "- Interpret the song/artist correctly from what sir said:\n"
        "    'Not Like Us' is a diss track by Kendrick Lamar (not Marina and the Diamonds).\n"
        "    When sir says an artist name, search for it exactly.\n"
        "- Use descriptive queries: 'Kendrick Lamar Not Like Us' not just 'Not Like Us'.\n"
        "- stop_media takes NO arguments — call it as stop_media() with empty params.\n"
        "- NEVER add spaces inside tool names."
    )

    def _run(self, task: Task) -> str:
        return self._brain.process_scoped(
            self._build_instruction(task),
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )
