from pathlib import Path

import MDAnalysis as mda
import numpy as np

from easybfe.analysis.trajectory import post_process_trajectory


def test_post_process_trajectory_includes_nearest_waters(tmp_path: Path):
    universe = mda.Universe.empty(
        7,
        n_residues=3,
        atom_resindex=np.array([0, 1, 1, 1, 2, 2, 2]),
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["C", "O", "H1", "H2", "O", "H1", "H2"])
    universe.add_TopologyAttr("resnames", ["MOL", "WAT", "WAT"])
    universe.add_TopologyAttr("resids", [1, 2, 3])
    universe.add_TopologyAttr("bonds", [(1, 2), (1, 3), (4, 5), (4, 6)])

    coordinates = np.array(
        [
            [
                [0.5, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [9.8, 0.0, 0.0],
                [9.0, 0.8, 0.0],
                [5.5, 0.0, 0.0],
                [5.5, 0.8, 0.0],
                [5.5, 0.0, 0.8],
            ],
            [
                [0.5, 0.0, 0.0],
                [5.5, 0.0, 0.0],
                [5.5, 0.8, 0.0],
                [5.5, 0.0, 0.8],
                [8.5, 0.0, 0.0],
                [9.3, 0.0, 0.0],
                [8.5, 0.8, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    dimensions = np.tile([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], (2, 1))
    universe.load_new(coordinates, dimensions=dimensions)

    input_pdb = tmp_path / "input.pdb"
    input_dcd = tmp_path / "input.dcd"
    universe.trajectory[0]
    universe.atoms.write(input_pdb)
    with mda.Writer(str(input_dcd), n_atoms=universe.atoms.n_atoms) as writer:
        for _ in universe.trajectory:
            writer.write(universe.atoms)

    output_pdb = tmp_path / "processed.pdb"
    output_dcd = tmp_path / "processed.dcd"
    post_process_trajectory(
        input_pdb,
        input_dcd,
        output_pdb,
        output_dcd,
        process_pbc=False,
        do_alignment=False,
        output_selection="resname MOL",
        include_water_selection="resname MOL",
        water_distance=2.0,
    )

    processed = mda.Universe(output_pdb, output_dcd)
    assert processed.atoms.n_atoms == 4
    assert len(processed.residues) == 2
    # With process_pbc=False the nearest water is shifted into the image of the target
    # (MOL at x=0.5). The frame-0 water oxygen has raw coordinate x=9.0 (min image
    # distance 1.5 to MOL); it is translated by -box (x -> -1.0) to stay adjacent. The
    # frame-1 nearest water oxygen at x=8.5 (min image distance 2.0) is moved to x=-1.5.
    np.testing.assert_allclose(processed.trajectory[0].positions[1], [-1.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(processed.trajectory[1].positions[1], [-1.5, 0.0, 0.0], atol=1e-3)

    # The dumped (no minimum image convention) water-target distance must equal the
    # minimum image distance, i.e. stay within the retention cutoff.
    for frame_index in range(len(processed.trajectory)):
        processed.trajectory[frame_index]
        mol_pos = processed.select_atoms("resname MOL").positions[0]
        water_o = processed.select_atoms("resname WAT").positions[0]
        raw_distance = np.linalg.norm(water_o - mol_pos)
        assert raw_distance <= 2.0 + 1e-3
    # Output serials: 1 = MOL carbon, 2 = water O, 3/4 = water H. Retained-water CONECT
    # records must stay within the water residue (only the regenerated O-H bonds); no
    # bond may link a water to the solute or to an atom outside the water residue.
    water_serials = {2, 3, 4}
    conect: dict[int, list[int]] = {}
    for line in output_pdb.read_text().splitlines():
        if line.startswith("CONECT"):
            serials = [
                int(line[index:index + 5])
                for index in range(6, len(line.rstrip()), 5)
                if line[index:index + 5].strip()
            ]
            if serials:
                conect[serials[0]] = serials[1:]
    for serial in water_serials:
        for partner in conect.get(serial, []):
            assert partner in water_serials
    assert not set(conect.get(1, [])) & water_serials
