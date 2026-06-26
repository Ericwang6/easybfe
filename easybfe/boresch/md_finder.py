"""Trajectory-based Boresch restraint finder (RXRX-MD pipeline)."""

import os
import math
import logging
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from ..core import Protein, Ligand
from .base import BoreschRestraintsFinder, BORESCH_FINDER_REGISTRY
from .restraint import BoreschRestraint
from .select_rep import select_representative_frame
from .utils import (
    _bond_series,
    _angle_series,
    _dihedral_series,
    _circular_mean_deg,
    _circular_std_rad,
    _map_ligand_atom_to_candidate,
    _enumerate_backbone_candidates,
    draw_ligand_anchors,
)


logger = logging.getLogger(__name__)


@BORESCH_FINDER_REGISTRY.register("rxrx-md")
class RxRxMDBoreschRestraintsFinder(BoreschRestraintsFinder):
    """Trajectory-based Boresch restraint finder (RXRX-MD pipeline).

    Implements the restraint-search procedure of Wu et al. 2025 (RXRX protocol).
    Protein-ligand hydrogen bonds and salt bridges are identified with PLIP over
    an MD trajectory (reusing :mod:`easybfe.analysis.interaction_plip`);
    interaction occupancies are aggregated per ligand candidate atom and protein
    residue, six-atom Boresch candidates are enumerated, filtered by a
    ``[angle_min, angle_max]`` window on the trajectory-mean angles, restricted to
    the ligand anchor closest to the ligand center of mass, and ranked by a
    standard-deviation-based score.

    This finder operates on an MD trajectory only: both ``topology`` and
    ``trajectory`` are required. The per-degree-of-freedom statistics are computed
    over the trajectory frames, and the PLIP interactions come either from an
    explicit precomputed ``interaction_csv`` or are generated on the fly from the
    ``topology`` / ``trajectory``.

    Parameters
    ----------
    protein : easybfe.core.Protein
        Protein whose :meth:`~easybfe.core.protein.Protein.to_openmm` ordering
        defines the protein atom/residue indices used for the anchors.
    ligand : easybfe.core.Ligand
        Ligand whose RDKit molecule defines the ligand atom indices.
    wts : tuple of float
        Six Boresch force constants (bond, two angles, three dihedrals).
    topology : os.PathLike
        Complex trajectory topology (e.g. ``prod_processed.pdb``). Required for
        PLIP-based interaction detection and for the frame source.
    trajectory : os.PathLike
        Complex trajectory (e.g. ``prod_processed.xtc``). Used to generate the
        interaction CSV (when not provided) and to compute the per-degree-of-
        freedom statistics.
    interaction_csv : os.PathLike, optional
        Precomputed PLIP interaction CSV (the :func:`easybfe.analysis.\
interaction_plip.analyze_multiple_frames` schema). When ``None`` it is generated
        from ``topology`` / ``trajectory``.
    ligand_residue_name : str, optional
        Residue name identifying the ligand for PLIP and the default ligand
        selection. Default is ``'MOL'``.
    ligand_selection : str, optional
        MDAnalysis selection for the ligand atoms. When ``None`` the
        ``'resname <ligand_residue_name>'`` selection is used, falling back to the
        first ``N`` atoms (``N`` = number of ligand atoms).
    protein_selection : str, optional
        MDAnalysis selection for protein atoms. Its atom order must match
        ``protein.to_openmm()``. Default is ``'protein'``.
    occupancy_threshold : float, optional
        Minimum aggregated interaction occupancy for a ligand-atom/residue pair to
        seed candidate enumeration. Default is ``0.5``.
    angle_min, angle_max : float, optional
        Allowed range (degrees) for the trajectory-mean Boresch angles. Defaults
        are ``45.0`` and ``135.0``.
    sin_power : float, optional
        Exponent applied to ``sin(theta_A) * sin(theta_B)`` in the score
        denominator. Default is ``2.0``.
    interaction_types : sequence of str, optional
        PLIP interaction types to consider. Default is
        ``('hydrogen_bond', 'salt_bridge')``.
    use_mpi, use_strict_hbond, remove_tmp : bool, optional
        Forwarded to :func:`easybfe.analysis.interaction_plip.analyze_interactions_for_trajectory` when generating the CSV.
    resnr_renum : dict, optional
        Optional residue-number remapping forwarded to PLIP.
    topology_format, trajectory_format : str, optional
        Format hints forwarded to MDAnalysis.
    workdir : os.PathLike, optional
        Working directory in which the finder stores its results: the
        representative protein/ligand structures, the representative-frame
        selection report and torsion-distribution plot, the Boresch candidate
        report and the ligand-anchor depiction. When ``None`` no files are
        written. See :meth:`find` for the exact file names.

    Notes
    -----
    When ``workdir`` is set, :meth:`find` additionally selects a representative
    trajectory frame (the frame whose ligand rotatable-torsion geometry is
    closest to the trajectory average, via
    :func:`easybfe.boresch.select_rep.select_representative_frame`) and writes the
    following files into ``workdir``:

    - ``representative_protein.pdb`` / ``representative_ligand.sdf``: the
      representative protein and ligand structures.
    - ``representative_selection.txt``: per-torsion circular means and the
      selected-frame values.
    - ``torsion_distributions.png``: the torsion-distribution plot.
    - ``boresch_candidates.txt``: the full Boresch candidate selection table.
    - ``ligand_anchors.png``: the 2D depiction of the three ligand anchors.

    Notes
    -----
    The score follows Eq. 1 of Wu et al. 2025 implemented as

    .. math::

        \\mathrm{score} = \\frac{\\sigma_r\\,\\sigma_{\\alpha}\\,\\sigma_{\\theta}\\,
        \\sigma_{\\gamma}\\,\\sigma_{\\beta}\\,\\sigma_{\\phi}}
        {(\\sin\\bar{\\alpha}\\,\\sin\\bar{\\theta})^{p}}

    where the :math:`\\sigma` are trajectory standard deviations (``r`` in nm,
    angles/dihedrals as circular standard deviations in radians), :math:`\\alpha`
    and :math:`\\theta` are the two Boresch angles, and ``p`` is ``sin_power``.
    """

    def __init__(
        self,
        protein: Protein,
        ligand: Ligand,
        wts: tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        topology: Optional[os.PathLike] = None,
        trajectory: Optional[os.PathLike] = None,
        interaction_csv: Optional[os.PathLike] = None,
        ligand_residue_name: str = "MOL",
        ligand_selection: Optional[str] = None,
        protein_selection: str = "protein",
        occupancy_threshold: float = 0.5,
        angle_min: float = 45.0,
        angle_max: float = 135.0,
        sin_power: float = 2.0,
        interaction_types: Sequence[str] = ("hydrogen_bond", "salt_bridge"),
        use_mpi: bool = False,
        use_strict_hbond: bool = False,
        remove_tmp: bool = True,
        resnr_renum: Optional[dict] = None,
        topology_format: Optional[str] = None,
        trajectory_format: Optional[str] = None,
        workdir: Optional[os.PathLike] = None,
        *args,
        **kwargs,
    ):
        super().__init__(protein, ligand, wts, workdir=workdir, *args, **kwargs)
        self.topology = topology
        self.trajectory = trajectory
        self.interaction_csv = interaction_csv
        self.ligand_residue_name = ligand_residue_name
        self.ligand_selection = ligand_selection
        self.protein_selection = protein_selection
        self.occupancy_threshold = occupancy_threshold
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.sin_power = sin_power
        self.interaction_types = tuple(interaction_types)
        self.use_mpi = use_mpi
        self.use_strict_hbond = use_strict_hbond
        self.remove_tmp = remove_tmp
        self.resnr_renum = resnr_renum
        self.topology_format = topology_format
        self.trajectory_format = trajectory_format

    # ------------------------------------------------------------------
    # Frame resolution
    # ------------------------------------------------------------------
    def _select_ligand_atomgroup(self, universe):
        """Return the ligand :class:`MDAnalysis.AtomGroup` for ``universe``."""
        if self.ligand_selection is not None:
            return universe.select_atoms(self.ligand_selection)
        atomgroup = universe.select_atoms(f"resname {self.ligand_residue_name}")
        if atomgroup.n_atoms == 0:
            n_ligand = self.ligand.get_rdmol().GetNumAtoms()
            atomgroup = universe.atoms[:n_ligand]
        return atomgroup

    def _load_trajectory_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Load protein and ligand positions for every trajectory frame."""
        import MDAnalysis as mda

        universe = mda.Universe(
            str(self.topology),
            str(self.trajectory),
            topology_format=self.topology_format,
            format=self.trajectory_format,
        )
        ligand_ag = self._select_ligand_atomgroup(universe)
        protein_ag = universe.select_atoms(self.protein_selection)
        protein_frames: list[np.ndarray] = []
        ligand_frames: list[np.ndarray] = []
        for _ in universe.trajectory:
            protein_frames.append(protein_ag.positions.copy())
            ligand_frames.append(ligand_ag.positions.copy())
        if not protein_frames:
            raise ValueError(f"No frames found in trajectory: {self.trajectory}")
        return np.asarray(protein_frames, dtype=float), np.asarray(ligand_frames, dtype=float)

    # ------------------------------------------------------------------
    # Interaction handling
    # ------------------------------------------------------------------
    def _build_index_maps(self) -> tuple[dict[int, int], dict[int, int]]:
        """Map trajectory universe atom indices to ligand/protein local indices.

        Returns two dictionaries keyed by universe atom index (0-based, i.e.
        ``PLIP serial - 1``): one to the ligand-local index (``ligand`` RDKit
        ordering) and one to the protein-local index (``protein.to_openmm()``
        ordering).
        """
        import MDAnalysis as mda

        universe = mda.Universe(str(self.topology), topology_format=self.topology_format)
        ligand_ag = self._select_ligand_atomgroup(universe)
        protein_ag = universe.select_atoms(self.protein_selection)
        uidx_to_ligand = {int(uidx): local for local, uidx in enumerate(ligand_ag.indices)}
        uidx_to_protein = {int(uidx): local for local, uidx in enumerate(protein_ag.indices)}
        return uidx_to_ligand, uidx_to_protein

    @staticmethod
    def _parse_idx_field(field) -> list[int]:
        """Parse a PLIP ``ligand_idx`` / ``protein_idx`` field into 1-based ints.

        The field may be a single value, a comma-joined list (salt bridges and
        other atom-group interactions) or empty/``NaN``.
        """
        if field is None:
            return []
        if isinstance(field, float) and math.isnan(field):
            return []
        text = str(field).strip()
        if not text or text.lower() == "nan":
            return []
        result = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.append(int(float(part)))
            except ValueError:
                continue
        return result

    def _resolve_interaction_dataframe(self):
        """Return the PLIP interaction table, or ``None`` if no source exists."""
        if self.interaction_csv is not None:
            import pandas as pd

            return pd.read_csv(str(self.interaction_csv), index_col=0)

        if self.topology is not None and self.trajectory is not None:
            from ..analysis.interaction_plip import analyze_interactions_for_trajectory

            return analyze_interactions_for_trajectory(
                top=str(self.topology),
                trj=str(self.trajectory),
                out_csv="",
                top_format=self.topology_format,
                trj_format=self.trajectory_format,
                use_mpi=self.use_mpi,
                remove_tmp=self.remove_tmp,
                ligand_residue_name=self.ligand_residue_name,
                use_strict_hbond=self.use_strict_hbond,
                resnr_renum=self.resnr_renum or {},
            )
        return None

    def _interaction_candidates(self, dataframe, ligand_mol, protein_atoms) -> list[tuple]:
        """Build six-atom candidates from a PLIP interaction table."""
        import pandas as pd

        uidx_to_ligand, uidx_to_protein = self._build_index_maps()
        filtered = dataframe[dataframe["interaction"].isin(self.interaction_types)]

        occupancy: dict[tuple[int, int], float] = defaultdict(float)
        residue_map: dict[int, object] = {}
        for _, row in filtered.iterrows():
            ratio = row.get("ratio")
            if ratio is None or (isinstance(ratio, float) and pd.isna(ratio)):
                continue
            ratio = float(ratio)

            ligand_candidate = None
            for serial in self._parse_idx_field(row.get("ligand_idx")):
                local = uidx_to_ligand.get(serial - 1)
                if local is None:
                    continue
                candidate = _map_ligand_atom_to_candidate(ligand_mol, local)
                if candidate is not None:
                    ligand_candidate = candidate
                    break
            if ligand_candidate is None:
                continue

            residue = None
            for serial in self._parse_idx_field(row.get("protein_idx")):
                local = uidx_to_protein.get(serial - 1)
                if local is None:
                    continue
                residue = protein_atoms[local].residue
                break
            if residue is None:
                continue

            occupancy[(ligand_candidate, residue.index)] += ratio
            residue_map[residue.index] = residue

        kept = [
            (ligand_candidate, residue_map[residue_index])
            for (ligand_candidate, residue_index), value in occupancy.items()
            if value > self.occupancy_threshold
        ]
        candidates: list[tuple] = []
        for ligand_candidate, residue in kept:
            candidates.extend(
                _enumerate_backbone_candidates(residue, ligand_candidate, protein_atoms, ligand_mol)
            )
        return list(dict.fromkeys(candidates))

    # ------------------------------------------------------------------
    # Scoring and selection
    # ------------------------------------------------------------------
    def _degree_of_freedom_series(self, candidate, protein_frames, ligand_frames):
        """Return the six Boresch DOF time series, or ``None`` if degenerate."""
        bb_ca, bb_c, bb_n, l1, l2, l3 = candidate
        pos_p1 = protein_frames[:, bb_ca]
        pos_p2 = protein_frames[:, bb_c]
        pos_p3 = protein_frames[:, bb_n]
        pos_l1 = ligand_frames[:, l1]
        pos_l2 = ligand_frames[:, l2]
        pos_l3 = ligand_frames[:, l3]

        series = (
            _bond_series(pos_l1, pos_p1),
            _angle_series(pos_p1, pos_l1, pos_l2),
            _angle_series(pos_p2, pos_p1, pos_l1),
            _dihedral_series(pos_p1, pos_l1, pos_l2, pos_l3),
            _dihedral_series(pos_p2, pos_p1, pos_l1, pos_l2),
            _dihedral_series(pos_p3, pos_p2, pos_p1, pos_l1),
        )
        if any(not np.all(np.isfinite(values)) for values in series):
            return None
        return series

    def _select_best(
        self,
        candidates,
        protein_frames,
        ligand_frames,
        ligand_com,
        mean_ligand,
        protein_atoms=None,
        ligand_mol=None,
    ):
        """Pick the best candidate by COM closeness then lowest score.

        Returns a ``(protein_anchors, ligand_anchors, rst_vals)`` tuple or
        ``None`` when no candidate passes the angle filter.

        When ``protein_atoms`` and ``ligand_mol`` are provided, both the selected
        candidate and the rejected/non-selected candidates (anchor atoms,
        restraint values, per-degree-of-freedom standard deviations, final score
        and rejection reason) are logged at ``INFO`` level via
        :meth:`_log_candidates`.
        """
        evaluated = []
        rejected = []
        for candidate in candidates:
            series = self._degree_of_freedom_series(candidate, protein_frames, ligand_frames)
            if series is None:
                rejected.append(
                    {
                        "candidate": candidate,
                        "reason": "non-finite degree-of-freedom series (degenerate geometry)",
                    }
                )
                continue
            r, alpha, theta, gamma, beta, phi = series

            # alpha and theta are bond angles (range [0, 180]), not dihedrals,
            # so plain (non-circular) statistics are used for them.
            mean_alpha = float(np.mean(alpha))
            mean_theta = float(np.mean(theta))
            alpha_range = (float(np.min(alpha)), float(np.max(alpha)))
            theta_range = (float(np.min(theta)), float(np.max(theta)))
            angle_info = {
                "mean_alpha": mean_alpha,
                "mean_theta": mean_theta,
                "alpha_range": alpha_range,
                "theta_range": theta_range,
            }
            if not (self.angle_min <= mean_alpha <= self.angle_max):
                rejected.append(
                    {
                        "candidate": candidate,
                        "reason": f"alpha outside angle window [{self.angle_min}, {self.angle_max}]",
                        **angle_info,
                    }
                )
                continue
            if not (self.angle_min <= mean_theta <= self.angle_max):
                rejected.append(
                    {
                        "candidate": candidate,
                        "reason": f"theta outside angle window [{self.angle_min}, {self.angle_max}]",
                        **angle_info,
                    }
                )
                continue

            sigma_r = float(np.std(r)) / 10.0  # Angstrom -> nm
            # Plain std for the bond angles (converted to radians for unit
            # consistency with the circular dihedral sigmas).
            sigma_alpha = float(np.std(np.radians(alpha)))
            sigma_theta = float(np.std(np.radians(theta)))
            sigma_gamma = _circular_std_rad(gamma)
            sigma_beta = _circular_std_rad(beta)
            sigma_phi = _circular_std_rad(phi)
            denominator = abs(
                math.sin(math.radians(mean_alpha)) * math.sin(math.radians(mean_theta))
            ) ** self.sin_power
            denominator = max(denominator, 1e-12)
            score = (
                sigma_r * sigma_alpha * sigma_theta * sigma_gamma * sigma_beta * sigma_phi
            ) / denominator

            l1 = candidate[3]
            com_distance = float(np.linalg.norm(mean_ligand[l1] - ligand_com))
            rst_vals = (
                float(np.mean(r)),
                mean_alpha,
                mean_theta,
                _circular_mean_deg(gamma),
                _circular_mean_deg(beta),
                _circular_mean_deg(phi),
            )
            sigmas = (
                sigma_r,
                sigma_alpha,
                sigma_theta,
                sigma_gamma,
                sigma_beta,
                sigma_phi,
            )
            evaluated.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "com_distance": com_distance,
                    "rst_vals": rst_vals,
                    "sigmas": sigmas,
                    "mean_alpha": mean_alpha,
                    "mean_theta": mean_theta,
                    "l1": l1,
                }
            )

        if not evaluated:
            if protein_atoms is not None and ligand_mol is not None:
                report = self._log_candidates([], rejected, None, protein_atoms, ligand_mol)
                self._write_candidates_report(report)
            return None

        # Choose the ligand anchor closest to the center of mass, then the
        # lowest-scoring restraint that uses that anchor.
        closest_l1 = min(evaluated, key=lambda item: item["com_distance"])["l1"]
        subset = [item for item in evaluated if item["l1"] == closest_l1]
        best = min(subset, key=lambda item: item["score"])
        if protein_atoms is not None and ligand_mol is not None:
            report = self._log_candidates(evaluated, rejected, best, protein_atoms, ligand_mol)
            self._write_candidates_report(report)
        candidate = best["candidate"]
        return tuple(candidate[:3]), tuple(candidate[3:]), best["rst_vals"]

    def _write_candidates_report(self, report: str) -> None:
        """Write the Boresch candidate report to ``workdir`` when one is set."""
        if self.workdir is None:
            return
        out_path = self.workdir / "boresch_candidates.txt"
        with open(out_path, "w") as handle:
            handle.write(report + "\n")
        logger.info("Wrote Boresch candidate report to %s", out_path)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ligand_anchor_repr(candidate, ligand_mol) -> str:
        """Return a compact ``L1,L2,L3`` ligand-anchor string for a candidate."""

        def _lig(idx):
            atom = ligand_mol.GetAtomWithIdx(int(idx))
            info = atom.GetMonomerInfo()
            name = info.GetName().strip() if info is not None else ""
            return f"{name or atom.GetSymbol()}[{idx}]"

        return ",".join(_lig(candidate[i]) for i in (3, 4, 5))

    @staticmethod
    def _protein_anchor_repr(candidate, protein_atoms) -> str:
        """Return a compact ``P1,P2,P3`` protein-anchor string for a candidate."""

        def _pro(idx):
            atom = protein_atoms[idx]
            residue = atom.residue
            return f"{atom.name}@{residue.name}{residue.id}[{idx}]"

        return ",".join(_pro(candidate[i]) for i in (0, 1, 2))

    @staticmethod
    def _format_table(headers, aligns, rows) -> str:
        """Render ``rows`` as a fixed-width text table.

        Parameters
        ----------
        headers : sequence of str
            Column header labels.
        aligns : sequence of str
            Per-column alignment, ``'<'`` for left or ``'>'`` for right.
        rows : sequence of sequence of str
            Already-formatted cell strings, one inner sequence per row.

        Returns
        -------
        str
            The assembled table (header, separator and one line per row).
        """
        widths = [len(h) for h in headers]
        for row in rows:
            for col, cell in enumerate(row):
                widths[col] = max(widths[col], len(cell))

        def _fmt_row(cells):
            return "  ".join(
                f"{cell:{align}{width}}"
                for cell, align, width in zip(cells, aligns, widths)
            )

        sep = "  ".join("-" * width for width in widths)
        lines = [_fmt_row(headers), sep]
        lines.extend(_fmt_row(row) for row in rows)
        return "\n".join(lines)

    def _legend_text(self) -> str:
        """Return the legend describing the candidate table columns.

        The six Boresch degrees of freedom are defined in terms of the protein
        anchors ``a, b, c`` (= ``P1, P2, P3``) and the ligand anchors
        ``A, B, C`` (= ``L1, L2, L3``). Each table value is reported as
        ``mean(std)`` over the trajectory frames, with all distances in
        Angstrom and all angles/dihedrals in degrees.
        """
        return (
            "Boresch degree-of-freedom legend (each table value is mean(std) "
            "over the trajectory frames):\n"
            "  Anchor atoms: protein a,b,c = P1,P2,P3 (protein_anchors column "
            "order); ligand A,B,C = L1,L2,L3 (ligand_anchors column order).\n"
            "  r     = distance(a-A)       [A]\n"
            "  alpha = angle(a-A-B)        [deg]\n"
            "  theta = angle(b-a-A)        [deg]\n"
            "  gamma = dihedral(a-A-B-C)   [deg]\n"
            "  beta  = dihedral(b-a-A-B)   [deg]\n"
            "  phi   = dihedral(c-b-a-A)   [deg]\n"
            "  COM   = distance from ligand anchor A to the ligand center of "
            "mass [A].\n"
            "  score = (std_r * std_alpha * std_theta * std_gamma * std_beta * "
            f"std_phi) / (sin(alpha) * sin(theta))^{self.sin_power:g}; "
            "lower is better; '*' marks the selected candidate."
        )

    def _candidate_row(self, item, protein_atoms, ligand_mol, marker: str):
        """Build one table row (list of cell strings) for a scored candidate.

        Each degree-of-freedom cell fuses the mean value and its standard
        deviation as ``mean(std)``. Angle/dihedral standard deviations are
        converted from radians to degrees and the bond standard deviation from
        nm to Angstrom so the whole table shares consistent units.
        """
        candidate = item["candidate"]
        r0, alpha0, theta0, gamma0, beta0, phi0 = item["rst_vals"]
        sigma_r, sigma_a, sigma_th, sigma_g, sigma_b, sigma_ph = item["sigmas"]
        return [
            marker,
            self._ligand_anchor_repr(candidate, ligand_mol),
            self._protein_anchor_repr(candidate, protein_atoms),
            f"{r0:.2f}({sigma_r * 10.0:.2f})",
            f"{alpha0:.1f}({math.degrees(sigma_a):.2f})",
            f"{theta0:.1f}({math.degrees(sigma_th):.2f})",
            f"{gamma0:.1f}({math.degrees(sigma_g):.2f})",
            f"{beta0:.1f}({math.degrees(sigma_b):.2f})",
            f"{phi0:.1f}({math.degrees(sigma_ph):.2f})",
            f"{item['com_distance']:.2f}",
            f"{item['score']:.3e}",
            "selected" if marker == "*" else "passed",
        ]

    def _rejected_row(self, item, protein_atoms, ligand_mol):
        """Build one table row for a candidate filtered out before scoring.

        Only the mean ``alpha`` / ``theta`` angles (in degrees) are available
        for these candidates; the remaining numeric columns are filled with
        ``-``.
        """
        candidate = item["candidate"]
        mean_alpha = item.get("mean_alpha")
        mean_theta = item.get("mean_theta")
        return [
            " ",
            self._ligand_anchor_repr(candidate, ligand_mol),
            self._protein_anchor_repr(candidate, protein_atoms),
            "-",
            f"{mean_alpha:.1f}" if mean_alpha is not None else "-",
            f"{mean_theta:.1f}" if mean_theta is not None else "-",
            "-", "-", "-",
            "-",
            "-",
            item["reason"],
        ]

    def _build_candidates_report(self, evaluated, rejected, best, protein_atoms, ligand_mol) -> str:
        """Render every candidate as a single aligned dataframe-style table.

        Each Boresch candidate occupies one row, ordered as: the selected
        candidate (marked with ``*``), then the remaining scored candidates by
        ascending score, then the candidates filtered out by the angle window or
        a degenerate geometry. Angles and dihedrals (and their sigmas) are
        reported in degrees, ``r`` in Angstrom, ``sigma_r`` in nm and the COM
        distance in Angstrom. Filtered candidates only populate the columns that
        are defined for them (mean ``alpha`` / ``theta`` when available) and use
        ``-`` elsewhere.

        Parameters
        ----------
        evaluated : list of dict
            Candidates that passed the angle filter and were scored.
        rejected : list of dict
            Candidates that were filtered out, each with a ``'reason'`` key.
        best : dict or None
            The selected candidate, or ``None`` when none passed the filter.
        protein_atoms : list of openmm.app.topology.Atom
            All protein atoms, indexable by global atom index.
        ligand_mol : rdkit.Chem.Mol
            Ligand molecule with explicit hydrogens.

        Returns
        -------
        str
            The assembled report (summary line, legend and candidate table).
        """
        summary = (
            f"Boresch candidate selection: {len(evaluated) + len(rejected)} total, "
            f"{len(evaluated)} passed angle filter, {len(rejected)} rejected."
        )

        headers = [
            "sel", "ligand_anchors", "protein_anchors",
            "r", "alpha", "theta", "gamma", "beta", "phi",
            "COM", "score", "status",
        ]
        aligns = ["<", "<", "<"] + [">"] * 8 + ["<"]

        rows = []
        if best is not None:
            rows.append(self._candidate_row(best, protein_atoms, ligand_mol, "*"))
        others = [item for item in evaluated if item is not best]
        for item in sorted(others, key=lambda it: it["score"]):
            rows.append(self._candidate_row(item, protein_atoms, ligand_mol, " "))
        for item in rejected:
            rows.append(self._rejected_row(item, protein_atoms, ligand_mol))

        return "\n".join(
            [summary, self._legend_text(), self._format_table(headers, aligns, rows)]
        )

    def _log_candidates(self, evaluated, rejected, best, protein_atoms, ligand_mol) -> str:
        """Log the candidate report (see :meth:`_build_candidates_report`).

        Returns the rendered report text so callers may also persist it to disk.
        """
        report = self._build_candidates_report(
            evaluated, rejected, best, protein_atoms, ligand_mol
        )
        logger.info("%s", report)
        return report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find(
        self,
        protein_positions: Optional[np.ndarray] = None,
        ligand_positions: Optional[np.ndarray] = None,
    ) -> BoreschRestraint:
        if self.topology is None or self.trajectory is None:
            raise ValueError(
                "RxRxMDBoreschRestraintsFinder requires both 'topology' and "
                "'trajectory' to be set."
            )

        protein_frames, ligand_frames = self._load_trajectory_frames()

        protein_atoms = list(self.protein.to_openmm().topology.atoms())
        ligand_mol = self.ligand.get_rdmol()

        n_protein = len(protein_atoms)
        n_ligand = ligand_mol.GetNumAtoms()
        if protein_frames.shape[1] != n_protein:
            raise ValueError(
                f"Resolved protein frames have {protein_frames.shape[1]} atoms but "
                f"protein.to_openmm() has {n_protein}; atom ordering would be inconsistent."
            )
        if ligand_frames.shape[1] != n_ligand:
            raise ValueError(
                f"Resolved ligand frames have {ligand_frames.shape[1]} atoms but the "
                f"ligand has {n_ligand}; atom ordering would be inconsistent."
            )

        # Ligand center of mass from the trajectory-mean ligand geometry.
        masses = np.array([atom.GetMass() for atom in ligand_mol.GetAtoms()])
        mean_ligand = ligand_frames.mean(axis=0)
        ligand_com = np.average(mean_ligand, axis=0, weights=masses)

        # Candidates come from PLIP hydrogen bonds / salt bridges, either read
        # from a precomputed CSV or generated on the fly from the trajectory.
        dataframe = self._resolve_interaction_dataframe()
        candidates = self._interaction_candidates(dataframe, ligand_mol, protein_atoms)
        logger.info("Found %d Boresch candidates from PLIP interactions", len(candidates))

        best = None
        if candidates:
            best = self._select_best(
                candidates,
                protein_frames,
                ligand_frames,
                ligand_com,
                mean_ligand,
                protein_atoms=protein_atoms,
                ligand_mol=ligand_mol,
            )

        if best is None:
            raise ValueError(
                "Could not find suitable Boresch restraint anchor candidates from "
                "protein-ligand interactions. This means the ligand is positioned "
                "in a weird pose."
            )

        protein_anchors, ligand_anchors, rst_vals = best
        restr = BoreschRestraint(
            protein_anchors=protein_anchors,
            ligand_anchors=ligand_anchors,
            rst_wts=self.wts,
        )
        restr.rst_vals = rst_vals

        # Representative-frame structure selection. This always runs as part of
        # the restraint search so that the six Boresch degrees of freedom can be
        # reported on the representative structure; the structure / plot / report
        # files are only written when a working directory is set.
        rep_info = None
        try:
            rep_info = self._select_representative_structure(ligand_mol)
        except Exception as exc:  # pragma: no cover - selection is auxiliary
            logger.warning("Failed to select representative structure: %s", exc)

        if rep_info is not None:
            self._report_representative_dof(restr, protein_frames, ligand_frames, rep_info)

        if self.workdir is not None:
            anchor_png = self.workdir / "ligand_anchors.png"
            try:
                draw_ligand_anchors(ligand_mol, restr.ligand_anchors, str(anchor_png))
                logger.info("Wrote ligand anchor depiction to %s", anchor_png)
            except Exception as exc:  # pragma: no cover - depiction is auxiliary
                logger.warning("Failed to draw ligand anchors: %s", exc)

        return restr

    # ------------------------------------------------------------------
    # Result output
    # ------------------------------------------------------------------
    def _select_representative_structure(self, ligand_mol) -> dict:
        """Select the representative frame and (optionally) write its outputs.

        Reuses :func:`easybfe.boresch.select_rep.select_representative_frame` with
        the same ligand/protein selections as the restraint search. The
        representative protein PDB, ligand SDF, torsion-distribution plot and
        per-torsion selection report are written into ``workdir`` when one is
        set; otherwise only the in-memory selection result is returned (so the
        representative-frame degrees of freedom can still be reported).

        Returns
        -------
        dict
            The result dictionary of
            :func:`easybfe.boresch.select_rep.select_representative_frame`,
            notably the ``'rep_frame'`` index.
        """
        ligand_selection = self.ligand_selection
        if ligand_selection is None:
            ligand_selection = f"resname {self.ligand_residue_name}"

        common = dict(
            topology=self.topology,
            trajectory=self.trajectory,
            ligand_mol=ligand_mol,
            protein_selection=self.protein_selection,
            topology_format=self.topology_format,
            trajectory_format=self.trajectory_format,
        )
        if self.workdir is not None:
            common.update(
                out_pdb=self.workdir / "representative_protein.pdb",
                out_sdf=self.workdir / "representative_ligand.sdf",
                out_fig=self.workdir / "torsion_distributions.png",
                out_info=self.workdir / "representative_selection.txt",
            )
        try:
            return select_representative_frame(ligand_selection=ligand_selection, **common)
        except ValueError:
            # Fall back to the first-N-atoms ligand selection, matching the
            # behavior of ``_select_ligand_atomgroup``.
            return select_representative_frame(ligand_selection=None, **common)

    def _report_representative_dof(
        self, restr: BoreschRestraint, protein_frames, ligand_frames, rep_info: dict
    ) -> Optional[tuple]:
        """Report the six Boresch DOF evaluated on the representative structure.

        The selected anchors are evaluated on the representative trajectory frame
        (indexed into the already-loaded ``protein_frames`` / ``ligand_frames``)
        and the resulting degrees of freedom are logged alongside the
        trajectory-mean values stored on ``restr``. When a working directory is
        set the comparison is also appended to ``representative_selection.txt``.

        Parameters
        ----------
        restr : easybfe.boresch.restraint.BoreschRestraint
            The selected restraint, providing the anchors and the trajectory-mean
            degrees of freedom.
        protein_frames, ligand_frames : numpy.ndarray
            Per-frame protein / ligand positions, shapes ``(F, n_protein, 3)`` and
            ``(F, n_ligand, 3)``, in the ``protein.to_openmm()`` / RDKit orderings.
        rep_info : dict
            Representative-frame selection result containing ``'rep_frame'``.

        Returns
        -------
        tuple or None
            The six representative-frame degrees of freedom, or ``None`` when the
            representative frame index is out of range.
        """
        rep_frame = int(rep_info["rep_frame"])
        if not (0 <= rep_frame < protein_frames.shape[0]):
            logger.warning(
                "Representative frame %d out of range for %d loaded frames; "
                "skipping representative-structure DOF report.",
                rep_frame,
                protein_frames.shape[0],
            )
            return None

        # Evaluate the selected anchors on the representative frame without
        # touching the trajectory-mean values stored on ``restr``.
        rep_restr = BoreschRestraint(
            protein_anchors=restr.protein_anchors,
            ligand_anchors=restr.ligand_anchors,
            rst_wts=self.wts,
        )
        rep_vals = rep_restr.compute_rst_vals(
            protein_frames[rep_frame], ligand_frames[rep_frame]
        )

        report = self._format_dof_comparison(restr.rst_vals, rep_vals, rep_frame)
        logger.info("%s", report)

        if self.workdir is not None:
            out_path = self.workdir / "representative_selection.txt"
            with open(out_path, "a") as handle:
                handle.write("\n" + report + "\n")
            logger.info("Appended representative-structure DOF report to %s", out_path)

        return rep_vals

    def _format_dof_comparison(self, mean_vals, rep_vals, rep_frame: int) -> str:
        """Render the trajectory-mean vs representative-structure Boresch DOF table.

        Parameters
        ----------
        mean_vals : sequence of float
            Trajectory-mean degrees of freedom ``(r, alpha, theta, gamma, beta,
            phi)`` (``r`` in Angstrom, the rest in degrees).
        rep_vals : sequence of float
            The same six degrees of freedom evaluated on the representative
            structure.
        rep_frame : int
            Index of the representative frame.

        Returns
        -------
        str
            The assembled report (header and a three-column table).
        """
        labels = ["r [A]", "alpha [deg]", "theta [deg]", "gamma [deg]", "beta [deg]", "phi [deg]"]
        rows = [
            [label, f"{float(mean):.2f}", f"{float(rep):.2f}"]
            for label, mean, rep in zip(labels, mean_vals, rep_vals)
        ]
        header = (
            "Selected Boresch restraint degrees of freedom: trajectory mean vs "
            f"representative structure (frame {rep_frame}). The trajectory-mean "
            "values are used as the restraint reference values (rst_vals)."
        )
        table = self._format_table(
            ["dof", "traj_mean", "representative"], ["<", ">", ">"], rows
        )
        return header + "\n" + table
