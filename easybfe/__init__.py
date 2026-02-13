try:
    from ._version import __version__
except ImportError:
    __version__ = '0.0.0'

import warnings

# MDAnalysis: PDB writer/reader missing formalcharges
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Found no information for attr:\s*'formalcharges'.*",
    module=r"MDAnalysis\.coordinates\.PDB",
)

# MDAnalysis: XDR offsets cache mismatch (common on scratch/parallel FS)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Reload offsets from trajectory.*",
    module=r"MDAnalysis\.coordinates\.XDR",
)