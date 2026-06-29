"""Backward-compatible re-export of the ABFE CLI group.

The canonical definition now lives in :mod:`easybfe.abfe.cli`.
"""
from ..abfe.cli import abfe

__all__ = ["abfe"]
