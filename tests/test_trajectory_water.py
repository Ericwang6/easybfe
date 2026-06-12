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
    np.testing.assert_allclose(processed.trajectory[0].positions[1], [9.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(processed.trajectory[1].positions[1], [8.5, 0.0, 0.0], atol=1e-3)
    water_serials = {2, 3, 4}
    for line in output_pdb.read_text().splitlines():
        if line.startswith("CONECT"):
            serials = {
                int(line[index:index + 5])
                for index in range(6, len(line), 5)
                if line[index:index + 5].strip()
            }
            assert not serials & water_serials
