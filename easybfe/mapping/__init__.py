"""
Author: Eric Wang
Date: 10/07/2024

This package contains functions to generate atom mapping between two molecules
"""
from typing import Dict, Any
from .openfe import OpenFEAtomMapper
from .lazymcs import LazyMCSMapper


def load_mapper(algorithm: str, options: Dict[str, Any]):
    if algorithm == 'lazymcs':
        return LazyMCSMapper(**options)
    else:
        return OpenFEAtomMapper(algorithm, **options)