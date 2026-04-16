"""
integrations/spotify.py — Spotify Web API integration for KOBRA.

Uses spotipy with OAuth PKCE flow (no client secret required for personal use).
Token cached in CredentialStore (encrypted).

Requires:
  - SPOTIFY_CLIENT_ID in .env
  - SPOTIFY_REDIRECT_URI in config (default: http://localhost:8888/callback)
  - Spotify Developer app with redirect URI set to the above

Scopes:
  - user-read-playback-state
  - user-modify-playback-state
  - user-read-currently-playing
  - playlist-read-private
  - user-library-read
"""

import logging
import os

import config
from credential_store import CredentialStore
from integrations.base_integration import BaseIntegration, IntegrationError

logger = logging.getLogger(__name__)

SPOTIFY_SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "playlist-read-private "
    "user-library-read"
)


class NoActiveDeviceError(IntegrationError):
    """Raised when Spotify is not open / playing on any device."""
    pass


class SpotifyIntegration(BaseIntegration):
    SERVICE_NAME = "spotify"

    def __init__(self, credential_store: CredentialStore) -> None:
        self._creds_store = credential_store
        self._sp = None   # spotipy.Spotify client, built on first auth

    # ── Auth ───────────────────────────────────────────────────────────────────

    def ensure_authenticated(self) -> bool:
        """Build/refresh the spotipy client. Returns True if ready."""
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth
        except ImportError:
            logger.error("[SPOTIFY] spotipy not installed. Run: pip install spotipy")
            return False

        client_id = getattr(config, "SPOTIFY_CLIENT_ID", None) or os.getenv("SPOTIFY_CLIENT_ID")
        redirect_uri = getattr(config, "SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

        if not client_id:
            logger.error(
                "[SPOTIFY] SPOTIFY_CLIENT_ID not set. "
                "Add it to .env or config.py. "
                "Get it from developer.spotify.com → Your App → Settings."
            )
            return False

        try:
            # spotipy handles its own token cache — we point it at our credentials DB location
            cache_path = getattr(config, "SPOTIFY_TOKEN_CACHE", ".spotify_token_cache")
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret="",        # Not needed for PKCE personal use
                redirect_uri=redirect_uri,
                scope=SPOTIFY_SCOPES,
                cache_path=cache_path,
                open_browser=True,
            )
            self._sp = spotipy.Spotify(auth_manager=auth_manager)
            # Trigger token fetch / validation
            self._sp.current_user()
            logger.info("[SPOTIFY] Authenticated successfully.")
            return True
        except Exception as exc:
            logger.error("[SPOTIFY] Auth failed: %s", exc)
            self._sp = None
            return False

    def _client(self):
        self._require_auth()
        return self._sp

    def _get_active_device_id(self) -> str | None:
        devices = self._client().devices()
        device_list = devices.get("devices", [])
        if not device_list:
            return None
        # Prefer currently active device; fallback to first available
        for d in device_list:
            if d.get("is_active"):
                return d["id"]
        return device_list[0]["id"]

    # ── Actions ────────────────────────────────────────────────────────────────

    def play(self, query: str) -> str:
        """Search for a track/artist/album/playlist and start playback."""
        sp = self._client()

        device_id = self._get_active_device_id()
        if device_id is None:
            raise NoActiveDeviceError(
                "No active Spotify device found. Open Spotify on any device first, sir."
            )

        try:
            # Search across all types — pick the best match
            results = sp.search(q=query, limit=1, type="track,artist,album,playlist")

            # Priority: track > album > playlist > artist
            uri = None
            label = query

            tracks = results.get("tracks", {}).get("items", [])
            albums = results.get("albums", {}).get("items", [])
            playlists = results.get("playlists", {}).get("items", [])
            artists = results.get("artists", {}).get("items", [])

            if tracks:
                item = tracks[0]
                uri = item["uri"]
                label = f"{item['name']} by {item['artists'][0]['name']}"
                sp.start_playback(device_id=device_id, uris=[uri])
            elif albums:
                item = albums[0]
                uri = item["uri"]
                label = f"{item['name']} by {item['artists'][0]['name']}"
                sp.start_playback(device_id=device_id, context_uri=uri)
            elif playlists:
                item = playlists[0]
                uri = item["uri"]
                label = item["name"]
                sp.start_playback(device_id=device_id, context_uri=uri)
            elif artists:
                item = artists[0]
                uri = item["uri"]
                label = item["name"]
                sp.start_playback(device_id=device_id, context_uri=uri)
            else:
                return f"I couldn't find anything for '{query}' on Spotify, sir."

            return f"Playing {label} on Spotify."

        except NoActiveDeviceError:
            raise
        except Exception as exc:
            logger.error("[SPOTIFY] play failed: %s", exc)
            raise IntegrationError(f"Spotify playback failed: {exc}") from exc

    def pause(self) -> str:
        try:
            self._client().pause_playback()
            return "Paused."
        except Exception as exc:
            raise IntegrationError(f"Pause failed: {exc}") from exc

    def resume(self) -> str:
        try:
            device_id = self._get_active_device_id()
            self._client().start_playback(device_id=device_id)
            return "Resumed."
        except Exception as exc:
            raise IntegrationError(f"Resume failed: {exc}") from exc

    def skip(self) -> str:
        try:
            self._client().next_track()
            return "Skipped."
        except Exception as exc:
            raise IntegrationError(f"Skip failed: {exc}") from exc

    def previous(self) -> str:
        try:
            self._client().previous_track()
            return "Going back."
        except Exception as exc:
            raise IntegrationError(f"Previous track failed: {exc}") from exc

    def set_volume(self, percent: int) -> str:
        percent = max(0, min(100, percent))
        try:
            device_id = self._get_active_device_id()
            self._client().volume(percent, device_id=device_id)
            return f"Spotify volume set to {percent}%."
        except Exception as exc:
            raise IntegrationError(f"Volume set failed: {exc}") from exc

    def get_current_track(self) -> str:
        try:
            playback = self._client().current_playback()
            if not playback or not playback.get("item"):
                return "Nothing is playing on Spotify right now, sir."
            item = playback["item"]
            track = item["name"]
            artist = item["artists"][0]["name"]
            is_playing = playback.get("is_playing", False)
            state = "Playing" if is_playing else "Paused on"
            return f"{state}: {track} by {artist}."
        except Exception as exc:
            raise IntegrationError(f"Could not get current track: {exc}") from exc
