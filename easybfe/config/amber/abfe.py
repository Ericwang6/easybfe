"""Backward-compatible re-exports for ABFE configuration models.

The canonical definitions now live in :mod:`easybfe.abfe.config` and
:mod:`easybfe.boresch.config`.
"""
from ...abfe.config import AmberAbfeConfig
from ...boresch.config import BoreschRestraintGeneratorConfig

__all__ = ["AmberAbfeConfig", "BoreschRestraintGeneratorConfig"]
