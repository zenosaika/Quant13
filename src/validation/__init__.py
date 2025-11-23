"""
Validation module for trade proposals

Ensures trade proposals align with research theses and meet safety criteria
"""

from .thesis_validator import (
    ThesisValidator,
    ThesisMismatchError,
    validate_thesis_alignment
)

__all__ = [
    "ThesisValidator",
    "ThesisMismatchError",
    "validate_thesis_alignment"
]
