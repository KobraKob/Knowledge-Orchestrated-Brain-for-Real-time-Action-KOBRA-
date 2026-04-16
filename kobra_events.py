"""
kobra_events.py — Shared event bus between KOBRA voice pipeline and UI.

Two directions:
  voice → UI  : post_event() from anywhere, broadcast via WebSocket
  UI → voice  : ui_command_queue.get() in main loop
"""

import queue

# Text commands typed in the UI → picked up by main loop
ui_command_queue: queue.Queue = queue.Queue()
