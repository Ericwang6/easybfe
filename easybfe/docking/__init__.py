from .base import BaseDocking, compute_box_from_coordinates

try:
    from .vina import VinaDocking
except ImportError:
    VinaDocking = None  # type: ignore[misc, assignment]