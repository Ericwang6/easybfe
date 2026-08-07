"""Replica-exchange statistics parsed from AMBER remlog files.

``pmemd`` writes one remlog per H-REMD stage (``-remlog <name>.log``), i.e. a
leg's production stage leaves ``<leg>/<prod_prefix>.log``. Each exchange attempt
block lists one row per replica::

    # exchange        1
    # Rep#, Neibr#, Temp0, PotE(x_1), PotE(x_2), left_fe, right_fe, Success, Success rate (i,i+1)
         1    24    298.15 -20684.21**********   -430.31      0.00    F        0.00
         2     3    298.15 -20854.47 -20914.32      0.00    -10.37    T        2.00

Neighbour pairing alternates between odd and even exchanges, so a row is an
attempt for the pair ``(i, i+1)`` only when the ``Neibr#`` column equals
``i + 1``; the wrap-around row (``Rep# = N``, ``Neibr# = 1``) is bookkeeping
only and never exchanged in a 1D ladder. Acceptance rates are recomputed here
from the ``Success`` column rather than read from the running "Success rate"
column, so the numbers are well defined even for a truncated log.

Columns can run together in the fixed-width output (e.g.
``298.15-117604.09``), so rows are parsed from the ends: the first two fields
are the replica and neighbour indices and the last two are the success flag and
the running rate.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


def parse_remlog(log_file: os.PathLike) -> Optional[dict]:
    """Summarize neighbour-pair exchange acceptance from an AMBER remlog.

    Parameters
    ----------
    log_file : os.PathLike
        Path to the remlog written by ``pmemd -rem 3 -remlog ...``.

    Returns
    -------
    dict or None
        ``None`` when the file does not exist or holds no exchange rows.
        Otherwise a dictionary with:

        ``n_replicas``
            Number of replicas (lambda windows) in the ladder.
        ``n_exchanges``
            Number of exchange attempt blocks found in the log.
        ``exchange_rate``
            Mean acceptance rate over all neighbour pairs.
        ``exchange_rate_min`` / ``exchange_rate_max``
            Extremes over the neighbour pairs.
        ``exchange_rate_per_pair``
            Acceptance rate of pair ``(i, i+1)`` at index ``i - 1``; length
            ``n_replicas - 1``.
        ``exchange_attempts_per_pair``
            Number of attempts each pair rate is based on.
    """
    path = Path(log_file)
    if not path.is_file():
        return None

    n_exchanges = 0
    n_replicas = 0
    attempts: dict[int, list[int]] = {}

    with path.open("r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.lstrip().startswith("#"):
                if "exchange" in line:
                    n_exchanges += 1
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                replica = int(fields[0])
                neighbor = int(fields[1])
            except ValueError:
                continue
            n_replicas = max(n_replicas, replica, neighbor)
            if neighbor != replica + 1:
                # Right-hand partner of the pair, or the (N, 1) wrap-around row.
                continue
            record = attempts.setdefault(replica, [0, 0])
            record[0] += 1
            record[1] += int(fields[-2].upper() == "T")

    if not attempts:
        logger.warning("No exchange records found in %s", path)
        return None

    rates: list[Optional[float]] = []
    counts: list[int] = []
    for replica in range(1, n_replicas):
        n_attempt, n_success = attempts.get(replica, [0, 0])
        counts.append(n_attempt)
        rates.append(n_success / n_attempt if n_attempt else None)

    finite = np.array([rate for rate in rates if rate is not None], dtype=float)
    return {
        "log_file": path.name,
        "n_replicas": int(n_replicas),
        "n_exchanges": int(n_exchanges),
        "exchange_rate": float(finite.mean()) if finite.size else None,
        "exchange_rate_min": float(finite.min()) if finite.size else None,
        "exchange_rate_max": float(finite.max()) if finite.size else None,
        "exchange_rate_per_pair": rates,
        "exchange_attempts_per_pair": counts,
    }
