"""
integrations/__init__.py — Service integration registry for KOBRA.

Each integration module is a self-contained class that wraps one external
service (Gmail, Calendar, Spotify, WhatsApp). They are instantiated once
at startup by IntegrationAgent and BrowserAgent.
"""
