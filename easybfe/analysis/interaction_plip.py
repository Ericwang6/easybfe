from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path
from io import StringIO
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import logging
from functools import partial
import xml.etree.ElementTree as ET

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

from plip.structure.preparation import PDBComplex, logger as PLIP_LOGGER
from plip.exchange.report import StructureReport
from plip.basic import config as PLIP_CONFIG

PLIP_LOGGER.setLevel(logging.ERROR)
PLIP_LOGGER.propagate = False

import MDAnalysis as mda


CONTACT_COLUMNS = [
    "interaction",
    "resname",
    "resnr",
    "chain",
    "ligand_idx",
    "protein_idx",
    "water_idx",
    "dist",
]


def _get_text(item: ET.Element, tag: str) -> Optional[str]:
    """Return the stripped text of ``item``'s child ``tag``, or ``None``."""
    el = item.find(tag)
    if el is None or el.text is None:
        return None
    text = el.text.strip()
    return text if text else None


def _get_idx_list(item: ET.Element, tag: str) -> Optional[str]:
    """
    Return a comma-joined string of atom indices contained in an atom-group
    element (``lig_idx_list`` / ``prot_idx_list``), or ``None`` if empty.
    """
    el = item.find(tag)
    if el is None:
        return None
    idxs = [c.text.strip() for c in el.findall("idx") if c.text and c.text.strip()]
    return ",".join(idxs) if idxs else None


def extract_contact(name: str, item: ET.Element) -> Dict[str, Optional[str]]:
    """
    Extract the ligand atom index, protein atom index, characteristic distance
    and (for water bridges) the bridging water index from a single PLIP
    interaction element.

    The atom indices correspond to the 1-based atom serial numbers of the input
    PDB file (PLIP's ``orig_idx``). For interactions that involve atom groups
    (salt bridges, pi-stacking and pi-cation interactions) the indices are
    returned as a comma-joined string. The characteristic distance is the
    distance PLIP uses to define each interaction type:

    - hydrophobic interaction: ligand-protein carbon distance (``dist``)
    - hydrogen bond: donor-acceptor distance (``dist_d-a``)
    - water bridge: the larger of the donor-water and acceptor-water distances
    - salt bridge / pi-cation: charge-center distance (``dist``)
    - pi-stacking: ring-center distance (``centdist``)
    - halogen bond: donor-acceptor distance (``dist``)
    - metal complex: metal-target distance (``dist``)

    Parameters
    ----------
    name : str
        The interaction tag name, e.g. ``'hydrogen_bond'``.
    item : xml.etree.ElementTree.Element
        The XML element describing one interaction.

    Returns
    -------
    dict
        A dict with keys ``'ligand_idx'``, ``'protein_idx'``, ``'water_idx'``
        and ``'dist'``. Missing values are ``None``.
    """
    ligand_idx = protein_idx = water_idx = dist = None

    if name == "hydrophobic_interaction":
        ligand_idx = _get_text(item, "ligcarbonidx")
        protein_idx = _get_text(item, "protcarbonidx")
        dist = _get_text(item, "dist")
    elif name == "hydrogen_bond":
        protisdon = _get_text(item, "protisdon") == "True"
        donor = _get_text(item, "donoridx")
        acceptor = _get_text(item, "acceptoridx")
        protein_idx, ligand_idx = (donor, acceptor) if protisdon else (acceptor, donor)
        dist = _get_text(item, "dist_d-a")
    elif name == "water_bridge":
        protisdon = _get_text(item, "protisdon") == "True"
        donor = _get_text(item, "donor_idx")
        acceptor = _get_text(item, "acceptor_idx")
        protein_idx, ligand_idx = (donor, acceptor) if protisdon else (acceptor, donor)
        water_idx = _get_text(item, "water_idx")
        candidates = [_get_text(item, "dist_a-w"), _get_text(item, "dist_d-w")]
        candidates = [float(c) for c in candidates if c is not None]
        dist = max(candidates) if candidates else None
    elif name == "salt_bridge":
        ligand_idx = _get_idx_list(item, "lig_idx_list")
        protein_idx = _get_idx_list(item, "prot_idx_list")
        dist = _get_text(item, "dist")
    elif name == "pi_stack":
        ligand_idx = _get_idx_list(item, "lig_idx_list")
        protein_idx = _get_idx_list(item, "prot_idx_list")
        dist = _get_text(item, "centdist")
    elif name == "pi_cation_interaction":
        ligand_idx = _get_idx_list(item, "lig_idx_list")
        protein_idx = _get_idx_list(item, "prot_idx_list")
        dist = _get_text(item, "dist")
    elif name == "halogen_bond":
        ligand_idx = _get_text(item, "don_idx")
        protein_idx = _get_text(item, "acc_idx")
        dist = _get_text(item, "dist")
    elif name == "metal_complex":
        metal_idx = _get_text(item, "metal_idx")
        target_idx = _get_text(item, "target_idx")
        location = (_get_text(item, "location") or "").lower()
        if location.startswith("protein") or location == "water":
            protein_idx, ligand_idx = metal_idx, target_idx
        else:
            ligand_idx, protein_idx = metal_idx, target_idx
        dist = _get_text(item, "dist")
    else:
        dist = _get_text(item, "dist")

    try:
        dist = float(dist) if dist is not None else None
    except (TypeError, ValueError):
        dist = None

    return {
        "ligand_idx": ligand_idx,
        "protein_idx": protein_idx,
        "water_idx": water_idx,
        "dist": dist,
    }


def analyze_single_frame(
    pdbpath: os.PathLike, 
    add_hydrogen: bool = False,
    resnr_renum: Dict[str, str] = dict(),
    write_xml: bool = True,
    ligand_residue_name: str = 'MOL',
    use_strict_hbond: bool = False
) -> List[Dict]:
    """
    Analyze a single ligand-complex structure.

    Parameters
    ----------
    pdbpath : os.PathLike
        Path to the pdbfile to be analyzed.
    add_hydrogen : bool
        Whether to add hydrogen to the pdb structure. If the PDB file already
        contains correct hydrogen information, set this to ``False``. Otherwise,
        set it to ``True``. Default is ``False``.
    resnr_renum : dict
        Optional mapping used to renumber protein residue numbers in the output.
    write_xml : bool
        Whether to dump the PLIP XML report next to ``pdbpath``.
    ligand_residue_name : str
        The residue name (``longname``) identifying the ligand binding site.
    use_strict_hbond : bool
        Use a stricter hydrogen-bond distance cutoff (3.5 A).

    Returns
    -------
    list of dict
        One record per detected ligand-protein contact. Each record contains the
        keys listed in :data:`CONTACT_COLUMNS`: ``interaction`` (the PLIP
        interaction type, e.g. ``'hydrogen_bond'``), ``resname``, ``resnr`` and
        ``chain`` of the protein residue, the ``ligand_idx`` and ``protein_idx``
        atom serial numbers, an optional bridging ``water_idx`` and the
        characteristic ``dist`` (see :func:`extract_contact`). Records are unique
        per atom pair within a frame.
    """
    
    if add_hydrogen:
        PLIP_CONFIG.NOHYDRO = False
    else:
        PLIP_CONFIG.NOHYDRO = True
    
    if use_strict_hbond:
        PLIP_CONFIG.HBOND_DIST_MAX = 3.5
        
    pdb = PDBComplex()
    pdb.load_pdb(str(pdbpath))
    pdb.analyze()
    report = StructureReport(pdb)

    tmp = sys.stdout
    xmlstr = StringIO()
    sys.stdout = xmlstr
    report.write_xml(True)
    sys.stdout = tmp
    xmlstr.seek(0)
    xmlstr = xmlstr.read()

    xmlobj = ET.fromstring(xmlstr)

    binding_sites = xmlobj.findall("./bindingsite")
    bs = [bs for bs in binding_sites if bs.findall("identifiers/longname")[0].text == ligand_residue_name][0]
    itypes = bs.findall("interactions/")
    contacts = []
    seen = set()
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            resnr = resnr_renum.get(resnr, resnr)
            chain = item.find("reschain").text
            indices = extract_contact(name, item)
            key = (name, resnr, chain, indices["ligand_idx"], indices["protein_idx"], indices["water_idx"])
            if key in seen:
                continue
            seen.add(key)
            contacts.append({
                "interaction": name,
                "resname": restype,
                "resnr": resnr,
                "chain": chain,
                "ligand_idx": indices["ligand_idx"],
                "protein_idx": indices["protein_idx"],
                "water_idx": indices["water_idx"],
                "dist": indices["dist"],
            })
    if write_xml:
        f_xml = Path(pdbpath).with_suffix('.xml')
        with open(f_xml, 'w') as f:
            f.write(xmlstr)
    return contacts


def analyze_multiple_frames(
    pdbpaths: List[os.PathLike], 
    f_csv: os.PathLike = '',
    use_mpi: bool = True, 
    chunksize: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Analyze multiple frames and write the aggregated results to a csv file.

    Each row of the returned table describes a unique ligand-protein contact
    (a specific atom pair) detected across the trajectory. The table contains
    the per-atom-pair columns produced by :func:`analyze_single_frame`
    (``interaction``, ``resname``, ``resnr``, ``chain``, ``ligand_idx``,
    ``protein_idx``, ``water_idx``) together with:

    - ``dist`` : mean characteristic distance over the frames in which the
      contact is present.
    - ``ratio`` : fraction of frames in which this specific atom pair is
      present.
    - ``residue_ratio`` : fraction of frames in which the residue forms the
      given interaction type through *any* atom pair (used for residue-level
      plotting).

    Parameters
    ----------
    pdbpaths : list of os.PathLike
        Per-frame PDB files to analyze.
    f_csv : os.PathLike
        Optional output path for the aggregated CSV report.
    use_mpi : bool
        Whether to parallelize the per-frame analysis with multiprocessing.
    chunksize : int
        Chunk size for the multiprocessing pool (``'auto'`` distributes evenly).
    **kwargs
        Forwarded to :func:`analyze_single_frame`.

    Returns
    -------
    pandas.DataFrame
        The aggregated interaction table.
    """
    interacts_frames = []
    analyze_single_frame_func = partial(
        analyze_single_frame, 
        **kwargs
    )
    if not use_mpi:
        for pdbpath in tqdm(pdbpaths, desc="Analyzing interactions"):
            frame_data = analyze_single_frame_func(pdbpath)
            interacts_frames.append(frame_data)
    else:
        import multiprocessing as mp
        import math
        num_cores = mp.cpu_count() - 2
        chunksize = math.ceil(len(pdbpaths) / num_cores) if chunksize == "auto" else int(chunksize)
        pool = mp.Pool(processes=num_cores)
        for frame_data in tqdm(
            pool.imap(func=analyze_single_frame_func, iterable=pdbpaths, chunksize=chunksize),
            total=len(pdbpaths), desc='Analyzing interactions'
        ):
            interacts_frames.append(frame_data)

    n_frames = len(pdbpaths)
    atom_agg = {}
    residue_count = {}
    for frame_data in interacts_frames:
        residues_seen = set()
        for rec in frame_data:
            res_key = (rec["interaction"], rec["resname"], rec["resnr"], rec["chain"])
            residues_seen.add(res_key)
            atom_key = res_key + (rec["ligand_idx"], rec["protein_idx"], rec["water_idx"])
            entry = atom_agg.setdefault(atom_key, {"count": 0, "dist_sum": 0.0, "dist_n": 0})
            entry["count"] += 1
            if rec["dist"] is not None:
                entry["dist_sum"] += rec["dist"]
                entry["dist_n"] += 1
        for res_key in residues_seen:
            residue_count[res_key] = residue_count.get(res_key, 0) + 1

    columns = ["interaction", "resname", "resnr", "chain", "ligand_idx",
               "protein_idx", "water_idx", "dist", "ratio", "residue_ratio"]
    rows = []
    for atom_key, entry in atom_agg.items():
        name, resname, resnr, chain, ligand_idx, protein_idx, water_idx = atom_key
        mean_dist = entry["dist_sum"] / entry["dist_n"] if entry["dist_n"] else float("nan")
        res_key = (name, resname, resnr, chain)
        rows.append({
            "interaction": name,
            "resname": resname,
            "resnr": resnr,
            "chain": chain,
            "ligand_idx": ligand_idx,
            "protein_idx": protein_idx,
            "water_idx": water_idx,
            "dist": round(mean_dist, 3) if mean_dist == mean_dist else mean_dist,
            "ratio": entry["count"] / n_frames if n_frames else float("nan"),
            "residue_ratio": residue_count[res_key] / n_frames if n_frames else float("nan"),
        })
    interact_df = pd.DataFrame(rows, columns=columns)
    if f_csv:
        interact_df.to_csv(str(f_csv))
    return interact_df


def analyze_interactions_for_trajectory(
    top: os.PathLike,
    trj: os.PathLike,
    tmp_dir: os.PathLike = '',
    out_csv: str = '',
    top_format: Optional[str] = None,
    trj_format: Optional[str] = None,
    use_mpi: bool = True,
    remove_tmp: bool = False,
    **kwargs
) -> pd.DataFrame:

    u = mda.Universe(top, trj, topology_format=top_format, format=trj_format)

    tmp_dir = tmp_dir if tmp_dir else os.path.join(os.path.dirname(trj), 'traj_pdbs')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    
    pdbs = []
    for i in tqdm(range(len(u.trajectory)), desc='Spliting trajectory to PDBs'):
        ts = u.trajectory[i]
        f_pdb = os.path.join(tmp_dir, f'{i}.pdb')
        u.atoms.write(f_pdb)
        pdbs.append(f_pdb)
    
    try:
        df = analyze_multiple_frames(pdbs, out_csv, use_mpi, **kwargs)
    finally:
        if remove_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return df
    


INTERACT_COLOR_MAP = {
    "hydrogen_bond": "C0",
    "hydrophobic_interaction": "C1",
    "pi_stack": "C2",
    "salt_bridge": "C3",
    "pi_cation_interaction": "C4",
    "halogen_bond": "C5",
    "water_bridge": "C6",
}


def plot_interactions(
    data: Union[os.PathLike, pd.DataFrame], 
    threshold: float = 0.1, 
    title: str = '',
    ax: Optional[Axes] = None,
    save_path: os.PathLike = '',
    skip_hydrophobic: bool = False,
    **kwargs
):

    if isinstance(data, str) or isinstance(data, Path):
        df = pd.read_csv(str(data), index_col=0)
    else:
        df = data

    ratio_col = 'residue_ratio' if 'residue_ratio' in df.columns else 'ratio'

    newdf = pd.DataFrame()
    df = df.sort_values(['resnr', 'chain'])
    df.index = list(range(df.shape[0]))
    for i in range(df.shape[0]):
        resname = df.loc[i, 'resname']
        chain = df.loc[i, 'chain']
        resnr = df.loc[i, 'resnr']
        restag = f"{resname}{resnr}{chain}"
        itype = df.loc[i, 'interaction']
        value = df.loc[i, ratio_col]
        # The detailed report may contain several atom-pair rows per residue and
        # interaction type; keep the strongest occupancy for the residue-level bar.
        current = 0.0
        if restag in newdf.index and itype in newdf.columns:
            existing = newdf.loc[restag, itype]
            if pd.notna(existing):
                current = existing
        newdf.loc[restag, itype] = max(current, value)

    newdf = newdf.fillna(0.0)
    newdf[newdf < threshold] = 0.0
    newdf = newdf.loc[(newdf != 0).any(axis=1), :]
    ind = np.arange(newdf.shape[0])

    bottom = np.zeros(newdf.shape[0])
    interacts = sorted(list(newdf.columns))

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(15, 5))
    
    for interact in interacts:
        if skip_hydrophobic and interact == 'hydrophobic_interaction':
            continue
        label = interact[:-12] if interact.endswith('interaction') else interact
        ax.bar(
            ind,
            newdf[interact],
            label=label,
            bottom=bottom,
            color=INTERACT_COLOR_MAP.get(interact, "C7"),
        )
        bottom += newdf[interact]
        for x, y, value in zip(ind, bottom, newdf[interact]):
            if value > 0:
                ax.text(x, y, f"{value:.2f}", ha='center', va='bottom')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(list(newdf.index), rotation=50)
    ax.legend()
    if title:
        ax.set_title(title)
    ax.set_ylim(0, max(1, np.max(bottom) * 1.1))
    ax.set_yticks([])
    if save_path:
        ax.figure.savefig(save_path, **kwargs)
    return ax
