"""Error hierarchy for the llm package. Callers can catch LLMError to cover everything."""


class LLMError(Exception):
    """Base class for every error this package raises."""


class ProviderNotFoundError(LLMError):
    """Raised when get_client() is asked for a provider name that isn't registered."""


class MissingCredentialsError(LLMError):
    """Raised when a provider needs an API key and none was found in args or env."""


class ConfigurationError(LLMError):
    """Raised for invalid client configuration, e.g. provider="custom" without a base_url."""
