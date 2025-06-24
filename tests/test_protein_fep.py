import pytest
import json
import os
from easybfe.amber.prep_protein_fep import setup_protein_fep_workflow


def test_protein_fep():
    dirname = os.path.join(os.path.dirname(__file__), 'data')
    with open(os.path.join(dirname, 'config_protein_fep.json')) as f:
        config = json.load(f)

    setup_protein_fep_workflow(
        proteinA_pdb=os.path.join(dirname, 'atHMT.pdb'),
        proteinB_pdb=os.path.join(dirname, 'atHMT_V140T.pdb'),
        ligand_mol="",
        ligand_top="",
        wdir=os.path.join(os.path.dirname(__file__), '_test_protein_fep'),
        config=config
    )
