"""Custom exception classes for the Scripture Translation system.

These exceptions are raised throughout the codebase to provide
typed error handling and clear error context.
"""


class ScriptureTranslationError(Exception):
    """Base exception for all Scripture Translation system errors."""

    pass


class ModelNotInitializedError(ScriptureTranslationError):
    """Raised when a route/method is called before the model is loaded."""

    pass


class LanguageNotSupportedError(ScriptureTranslationError):
    """Raised when a language code is not in Config.LANGUAGE_CODES."""

    def __init__(self, code: str) -> None:
        """Initialize with language code.

        Args:
            code: The unsupported language code that was requested.
        """
        super().__init__(f"Unsupported language code: {code!r}")
        self.code = code


class TermConflictError(ScriptureTranslationError):
    """Raised on unresolved terminology conflict when override=False."""

    pass


class TerminologyDBError(ScriptureTranslationError):
    """Raised for database access failures in TerminologyDB."""

    pass


class DataLoadError(ScriptureTranslationError):
    """Raised when Bible data cannot be loaded or parsed."""

    pass


class EvaluationError(ScriptureTranslationError):
    """Raised when an evaluation metric cannot be computed."""

    pass
