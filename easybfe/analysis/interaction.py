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


def analyze_single_frame(
    pdbpath: os.PathLike, 
    add_hydrogen: bool = False,
    resnr_renum: Dict[str, str] = dict(),
    write_xml: bool = True,
    ligand_residue_name: str = 'MOL',
    use_strict_hbond: bool = False
) -> Dict[str, int]:
    """
    Analyze a single ligand-complex structure

    Parameters
    ----------
    pdbpath: os.PathLike
        Path to the pdbfile to be analyzed
    add_hydrogen: bool
        Whether to add hydrogen to the pdb structure. If the PDB file already contains correct hydrogen information, set this to False. 
        Otherwise, set it to True. Default is False.
    
    Return
    ------
    interact_count_frame: Dict[str, int]
        A dict with interaction type as the key, and the number of interaction as value
        The key is in the format "{name}/{restype}/{resnr}/{chain}". For example,
        'hydrophobic_interaction/ALA/123/A'
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
    interact_count_frame = {}
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            resnr = resnr_renum.get(resnr, resnr)
            chain = item.find("reschain").text
            sig = f"{name}/{restype}/{resnr}/{chain}"
            cnt = interact_count_frame.get(sig, 0)
            if cnt == 0:
                interact_count_frame.update({sig: cnt+1})
    if write_xml:
        f_xml = Path(pdbpath).with_suffix('.xml')
        with open(f_xml, 'w') as f:
            f.write(xmlstr)
    return interact_count_frame


def analyze_multiple_frames(
    pdbpaths: List[os.PathLike], 
    f_csv: os.PathLike = '',
    use_mpi: bool = True, 
    chunksize: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Analyze multiple frames and write results to a csv file
    """
    interacts_frames = []
    analyze_single_frame_func = partial(
        analyze_single_frame, 
        **kwargs
    )
    if not use_mpi:
        for pdbpath in tqdm(pdbpaths):
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
                
    interacts = {}
    for frame_data in interacts_frames:
        for sig, cnt in frame_data.items():
            val = interacts.get(sig, 0)
            interacts.update({sig: val+cnt})
            
    interact_df = []
    for sig, val in interacts.items():
        name, resname, resnr, chain = tuple(sig.split("/"))
        ratio = val / len(pdbpaths)
        interact_df.append({
            "interaction": name,
            "resname": resname,
            "resnr": resnr,
            "chain": chain,
            "ratio": ratio
        })
    interact_df = pd.DataFrame(interact_df)
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
    
    df = analyze_multiple_frames(pdbs, out_csv, use_mpi, **kwargs)

    if remove_tmp:
        for p in pdbs:
            os.remove(p)

    return df
    


INTERACT_COLOR_MAP = {
    "hydrogen_bond": "C0",
    "hydrophobic_interaction": "C1",
    "pi_stack": "C2",
    "salt_bridge": "C3",
    "pi_cation_interaction": "C4",
    "halogen_bond": "C5"
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
        df = pd.read_csv(str(f_csv), index_col=0)
    else:
        df = data

    newdf = pd.DataFrame()
    df = df.sort_values(['resnr', 'chain'])
    df.index = list(range(df.shape[0]))
    for i in range(df.shape[0]):
        resname = df.loc[i, 'resname']
        chain = df.loc[i, 'chain']
        resnr = df.loc[i, 'resnr']
        restag = f"{resname}{resnr}{chain}"
        itype = df.loc[i, 'interaction']
        newdf.loc[restag, itype] = df.loc[i, 'ratio']

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
        ax.bar(ind, newdf[interact], label=label, bottom=bottom, color=INTERACT_COLOR_MAP[interact])
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

