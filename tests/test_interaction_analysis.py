from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from easybfe.analysis.interaction_plip import plot_interactions


def test_plot_interactions_supports_water_bridge(tmp_path: Path):
    data = pd.DataFrame(
        [
            {
                "interaction": "water_bridge",
                "resname": "ASP",
                "resnr": 10,
                "chain": "A",
                "ratio": 0.5,
            }
        ]
    )

    output = tmp_path / "interaction.png"
    ax = plot_interactions(data, save_path=output)

    assert output.is_file()
    plt.close(ax.figure)
