import os
from easybfe.amber import AmberRbfeProject


def test_plain_md():
    pwd = os.path.dirname(__file__)
    datadir = os.path.join(pwd, 'data')
    wdir = os.path.join(pwd, '_test_plain_md')

    proj = AmberRbfeProject(wdir, init=True)
    proj.add_protein(os.path.join(datadir, 'tyk2_pdbfixer.pdb'), name='tyk2')
    proj.add_ligand(os.path.join(datadir, 'jmc_23.sdf'), name='jmc_23', protein_name='tyk2', parametrize=True, charge_method='gas', overwrite=True)
    config = os.path.join(pwd, 'config_plain_md.json')
    proj.run_plain_md(
        protein_name="tyk2",
        ligand_name="jmc_23",
        task_name="jmc_23_complex",
        config=config
    )
    proj.run_plain_md(
        protein_name="tyk2",
        task_name="tyk2",
        config=config
    )
    proj.run_plain_md(
        protein_name="tyk2",
        ligand_name="jmc_23",
        task_name="jmc_23",
        config=config,
        ligand_only=True
    )

