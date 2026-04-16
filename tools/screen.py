"""
tools/screen.py — Vision-based screen understanding for KOBRA.

Uses Groq's llama-3.2-11b-vision-preview (free tier) to understand what's
on screen and find UI elements by natural language description.

No coordinates needed — describe the element and KOBRA finds and clicks it.
"""

import base64
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime

import config

logger = logging.getLogger(__name__)


# ── Core: screenshot + vision ─────────────────────────────────────────────────

def take_screenshot(region: str = "full") -> str:
    """
    Take a screenshot and save to a temp file.
    region: 'full' | 'terminal' (bottom 40%) | 'code' (middle 65%)
    Returns the temp file path.
    """
    import pyautogui
    screenshot = pyautogui.screenshot()

    if region == "terminal":
        w, h = screenshot.size
        screenshot = screenshot.crop((0, int(h * 0.6), w, h))
    elif region == "code":
        w, h = screenshot.size
        screenshot = screenshot.crop((0, int(h * 0.1), w, int(h * 0.75)))

    path = os.path.join(tempfile.gettempdir(), f"kobra_screen_{int(time.time() * 1000)}.png")
    screenshot.save(path)
    return path


def _send_to_vision(screenshot_path: str, question: str) -> str:
    """
    Send a screenshot to Groq's vision model and return the response.
    Uses llama-3.2-11b-vision-preview — available on free tier.
    Cleans up the temp file after sending.
    """
    from groq import Groq

    try:
        with open(screenshot_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return "Screenshot file not found."
    finally:
        try:
            os.remove(screenshot_path)
        except Exception:
            pass

    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                    {"type": "text", "text": question},
                ],
            }],
            max_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("Vision API call failed: %s", exc)
        return f"Vision analysis failed: {exc}"


# ── Public tool functions ─────────────────────────────────────────────────────

def read_screen(question: str = "Describe everything visible on screen.",
                region: str = "full") -> str:
    """Take a screenshot and answer a question about what's visible."""
    path = take_screenshot(region)
    return _send_to_vision(path, question)


def find_element_on_screen(element_description: str) -> dict | None:
    """
    Find a UI element by natural language description.
    Returns {"x": int, "y": int} pixel coordinates or None if not found.
    """
    import pyautogui
    screen_w, screen_h = pyautogui.size()
    path = take_screenshot("full")

    question = (
        f"Look at this screenshot (resolution: {screen_w}x{screen_h}). "
        f'Find: "{element_description}"\n\n'
        f"Return ONLY a JSON object with x and y pixel coordinates, or null if not found.\n"
        f'Example: {{"x": 523, "y": 187}}'
    )
    result = _send_to_vision(path, question)

    try:
        match = re.search(r'\{[^{}]*?"x"[^{}]*?\}', result, re.DOTALL)
        if match:
            coords = json.loads(match.group())
            if "x" in coords and "y" in coords:
                return {"x": int(coords["x"]), "y": int(coords["y"])}
    except Exception as exc:
        logger.debug("Coordinate parse failed: %s | raw: %s", exc, result[:100])
    return None


def click_element(element_description: str) -> str:
    """Find a UI element by description and click it."""
    coords = find_element_on_screen(element_description)
    if not coords:
        return f"Could not find '{element_description}' on screen."
    try:
        import pyautogui
        pyautogui.click(coords["x"], coords["y"])
        return f"Clicked '{element_description}' at ({coords['x']}, {coords['y']})."
    except Exception as exc:
        return f"Found element but click failed: {exc}"
