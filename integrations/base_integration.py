"""
integrations/base_integration.py — Abstract base class for all KOBRA service integrations.

Every integration:
  - Implements ensure_authenticated() to check / trigger OAuth / session login
  - Raises NotAuthenticatedError if auth fails — caught by agents for voice feedback
  - Implements _require_auth() which agents call at the top of every action method
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class NotAuthenticatedError(Exception):
    """Raised when a service is not authenticated and auth flow cannot complete."""
    def __init__(self, service: str) -> None:
        self.service = service
        super().__init__(f"Not authenticated with {service}. Please complete the login flow.")


class IntegrationError(Exception):
    """General integration error (network, API quota, unexpected response, etc.)."""
    pass


class BaseIntegration(ABC):
    SERVICE_NAME: str = ""

    @abstractmethod
    def ensure_authenticated(self) -> bool:
        """
        Check if valid credentials exist. If not, trigger the auth flow.
        Returns True if authenticated, False if auth could not be completed.
        Implementations should be idempotent — safe to call on every action.
        """
        ...

    def _require_auth(self) -> None:
        """
        Call at the top of every action method.
        Raises NotAuthenticatedError if authentication cannot be established.
        """
        if not self.ensure_authenticated():
            raise NotAuthenticatedError(self.SERVICE_NAME)
