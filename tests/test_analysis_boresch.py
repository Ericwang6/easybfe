from pathlib import Path

import numpy as np

from easybfe.analysis.boresch import parse_boresch_restraints


def test_parse_boresch_restraints(tmp_path: Path):
    input_path = tmp_path / "prod.in"
    input_path.write_text(
        """
&rst iat=1,2, r1=0.0, r2=3.5, r3=3.5, r4=99.0, /
&rst iat=2,1,3, r1=0.0, r2=100.0, r3=100.0, r4=180.0, /
&rst iat=4,2,1, r1=0.0, r2=110.0, r3=110.0, r4=180.0, /
&rst iat=2,1,3,5, r1=-240.0, r2=-60.0, r3=-60.0, r4=120.0, /
&rst iat=4,2,1,3, r1=-90.0, r2=90.0, r3=90.0, r4=270.0, /
&rst iat=6,4,2,1, r1=-150.0, r2=30.0, r3=30.0, r4=210.0, /
"""
    )

    atom_groups, targets = parse_boresch_restraints(input_path)

    assert atom_groups == [(0, 1), (1, 0, 2), (3, 1, 0), (1, 0, 2, 4), (3, 1, 0, 2), (5, 3, 1, 0)]
    np.testing.assert_allclose(targets, [3.5, 100.0, 110.0, -60.0, 90.0, 30.0])
