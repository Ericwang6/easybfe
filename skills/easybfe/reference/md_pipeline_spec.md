# MD Pipeline Config Specification

`easybfe md pipeline` reads a YAML or JSON config validated as `MDPipelineConfig`
(`easybfe.md.config`).

See [assets/config_md_1ns.yaml](../assets/config_md_1ns.yaml) for a complete working example
(verified end to end on TYK2 for all three system types).

## Top-Level Fields

| Field          | Type   | Default | Description |
| -------------- | ------ | ------- | ----------- |
| `protein`      | path   | —       | Protein PDB. Overridden by `-p`. |
| `ligand`       | path   | —       | Raw ligand file (SDF, ...), parameterized with `ligand_param`; or a parameterized ligand directory / `.ligpack`, used as is. Overridden by `-l`. |
| `output_dir`   | path   | —       | Output directory `<MD-DIR>`. Overridden by `-o`. Required (config or CLI). |
| `task_name`    | string | output dir name | Title on the RMSD / interaction plots. |
| `ligand_param` | object | gaff2 / bcc | Ligand parameterization (below). |
| `simulation`   | object | default 5-stage MD | System building + MD workflow (below). |
| `analysis`     | object | per system type | Trajectory analysis (below). |

At least one of `protein` / `ligand` is required. Which ones are given decides the system:

| Given              | System                         | `task_type` |
| ------------------ | ------------------------------ | ----------- |
| protein + ligand   | protein-ligand complex         | `complex`   |
| protein only       | apo protein                    | `protein`   |
| ligand only        | ligand in a water box          | `ligand`    |

## `ligand_param`

Same block as in `abfe pipeline`. Ignored when the ligand is already parameterized or absent.

| Field           | Default | Description |
| --------------- | ------- | ----------- |
| `forcefield`    | `gaff2` | `gaff`, `gaff2`, `openff-2.x.x`, or a path to an `.xml` |
| `charge_method` | `bcc`   | `bcc`, `gas`, `resp`, ... |
| `engine`        | `""`    | Force `acpype` / `openff` / `custom`; auto-detected when empty |
| `resp_engine`   | `""`    | e.g. `qchem`; only for `charge_method: resp*` |

## `simulation` — System and MD Workflow

Same fields as the `simulation` block of `easybfe md setup` (`AmberSimulationConfig`), and the
same system fields as the ABFE legs (see [abfe_setup_spec.md](abfe_setup_spec.md) for the full
`SetupConfig`, `workflow` and `cntrl` tables).

| Field            | Default   | Description |
| ---------------- | --------- | ----------- |
| `box_shape`      | `cube`    | `cube`, `dodecahedron`, `octahedron` |
| `buffer`         | `20.0`    | Solvent padding (Angstrom), OpenMM convention: box width = max(2·buffer, solute diameter + buffer) |
| `ionic_strength` | `0.15`    | NaCl (molar); system is also neutralized |
| `do_hmr`         | `true`    | Hydrogen mass repartitioning; needed for 4 fs steps |
| `protein_ff`     | `[ff14sb]`| Protein force field(s) |
| `water_ff` / `water_model` | `tip3p` | Water |
| `basename`       | `system`  | Name of `system.prmtop` / `.inpcrd` / `.pdb` |
| `workflow`       | 5 stages  | Ordered MD stages. **For plain MD set `use_remd: false` and `use_mpi: false` on every stage**, and use `exec: pmemd.cuda`. The last stage is the production that gets analyzed. |

`cntrl.num_steps` is accepted as an alias of `nstlim`, and `ofreq` sets the trajectory /
restart write stride, so frames = `num_steps / ofreq`.

**Small solutes:** the box is enlarged automatically (with a warning in `md.log`) when
it would be too small for pmemd. For `pmemd.cuda` it must fit 3 neighbor-list cells of
`cut + 2 Å` per dimension, i.e. ≥ 36 Å face to face for `cut: 10`. So a lone ligand
always gets at least that box, whatever `buffer` says.

## `analysis` — Trajectory Analysis

`PlainMDAnalysisConfig`. Fields you leave out get defaults for the system type; fields you
set always win.

| Field | complex | protein | ligand |
| ----- | ------- | ------- | ------ |
| `rmsd_selection`          | `resname MOL` | `backbone` | `resname MOL` |
| `align_selection`         | `backbone`    | `backbone` | `resname MOL` |
| `center_selection`        | `protein`     | `protein`  | `resname MOL` |
| `output_selection`        | `protein or resname MOL` | `protein or resname MOL` | `resname MOL` |
| `interaction_analysis`    | `true`  | `false` | `false` |
| `do_gbsa`                 | `true`  | `false` | `false` |
| `use_symmetry_correction` | `true`  | `false` | `true`  |

Other commonly changed fields:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `use_starting_structure_as_ref` | `true` | RMSD / alignment reference: the built system (`true`) or the first production frame (`false`) |
| `include_water_selection` | `null` | e.g. `resname MOL` to keep the waters near the ligand in `prod_processed.*` (complex only) |
| `water_distance` | `5.0` | Cutoff (Angstrom) for `include_water_selection` |
| `heavy_atoms_only` | `true` | RMSD on heavy atoms only |
| `gbsa_igb` / `gbsa_saltcon` / `gbsa_epsin` | `2` / `0.15` / `4.0` | GBSA model settings |

## Outputs

`<MD-DIR>` is a regular `md setup` directory, so `easybfe md analyze <MD-DIR>` also works on it:

```
<MD-DIR>/
├── md.log                  master log (setup, every stage's output, analysis, summary)
├── result.json             analysis summary (below)
├── config.json             resolved AmberPlainMDConfig
├── system.prmtop / .inpcrd / .pdb
├── protein.pdb, ligand/    inputs as used (whichever were given)
├── run.sh                  bash + AMBER only; resumable (run.sh --list / --from / --until)
├── status.json, pipeline_run.log
├── 01.em/ 02.nvt/ 03.npt/  equilibration stages
└── 04.prod/                production (last stage)
    ├── 04.prod.mdcrd, .out, .rst7, done.tag
    ├── prod_processed.pdb / .xtc     imaged + aligned, solvent stripped
    ├── prod_rmsd.txt / .png
    ├── interaction.csv / .png        complex only
    └── gbsa.txt                      complex only, kJ/mol per frame
```

`result.json`:

| Key | Description |
| --- | ----------- |
| `task_type` | `complex`, `protein` or `ligand` |
| `prod_stage` | Analyzed stage name |
| `trajectory` / `topology` | Processed trajectory files |
| `rmsd` | `selection`, `n_frames`, `simulated_ns`, `mean`, `std`, `max`, `final` (Angstrom) |
| `interaction_csv` | complex only |
| `gbsa` | complex only: `mean`, `std`, `n_frames` (kcal/mol). MM/GBSA end-point value, useful for ranking, not an absolute ΔG |

## Common Modifications

### Change production length

Production length = `num_steps × dt`. For 10 ns at 4 fs with 500 frames:

```yaml
  - type: prod
    name: 04.prod
    exec: pmemd.cuda
    use_remd: false
    use_mpi: false
    cntrl:
      dt: 0.004
      num_steps: 2500000
      ofreq: 5000
      efreq: 5000
```

### Restrain the protein backbone in production

```yaml
    cntrl:
      ntr: 1
      restraintmask: "@CA,C,N"
      restraint_wt: 5.0
```

### Use an already-parameterized ligand

Pass the directory or `.ligpack` from `easybfe ligand pargen`; `ligand_param` is then ignored:

```bash
easybfe md pipeline config_md_1ns.yaml -p protein.pdb -l ligands/jmc_23.ligpack -o run_complex
```
