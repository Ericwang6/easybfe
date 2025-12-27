import os, glob, time
from easybfe.gbsa import run_gbsa_for_ligand_conformers


def test_run_gbsa_for_ligand_conformers():
    start = time.time()
    gbsa_dir = os.path.join(os.path.dirname(__file__), 'data/gbsa_test_data')
    ligand_confs = list(glob.glob(os.path.join(gbsa_dir, '*.sdf')))
    run_gbsa_for_ligand_conformers(
        protein_pdb=os.path.join(gbsa_dir, 'sars_noh.pdb'),
        ligand_sdf=os.path.join(gbsa_dir, 'SARS-CoV-2_Mpro-P1701_0A_CONFIDENTIAL_frag4_conf7_opt.sdf'),
        ligand_confs=ligand_confs,
        wdir=os.path.join(gbsa_dir, '_test_gbsa_bcc'),
        charge_method='gas'
    )
    end = time.time()
    print(f'GBSA for ligand conformers took {end - start:.2f} seconds.')