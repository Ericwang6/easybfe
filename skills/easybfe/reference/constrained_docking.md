# Constrained Docking (`easybfe ligand cdock`)

Use constrained docking when you need probe ligands to preserve a known binding mode from a reference ligand, while still allowing local optimization in the receptor.

## Command

```bash
easybfe ligand cdock PROBE \
  --ref REF_SDF \
  --protein PROTEIN.{pdb|pdbqt} \
  [--output OUT.sdf | --output-dir OUT_DIR] \
  [OPTIONS]
```

Required positional/flags:

- `PROBE`: probe ligand file (all records are processed).
- `--ref/-r`: reference ligand SDF (first record is used; requires 3D conformer).
- `--protein/-p`: receptor structure (`.pdb` or `.pdbqt`).

Output rules:

- Use `--output/-o` for a single probe ligand.
- Use `--output-dir/-O` for multiple probes (writes one `name.sdf` per probe).
- If both are given, `--output-dir` is used.

## Options

### Docking region

- `--box-center X Y Z` + `--box-size X Y Z`: explicit docking box in Angstrom.
- Both must be provided together.
- If omitted, the box is inferred from the reference ligand.

### Pose refinement controls

- `--no-em`: skip OpenMM energy minimization with the protein.
- `--harmonic-restraints`: during EM, use harmonic restraints for mapped atoms instead of freezing them.
- `--restraint-k`: harmonic force constant in `kcal/mol/A^2` (default: `10.0`).

### Mapping and receptor prep

- `--mapping-json`: optional JSON mapping file with object format `{mol_atom_idx: ref_atom_idx}`.
- If omitted, mapping is generated automatically.
- `--protein-prep-exec`: receptor preparation executable (`obabel` or `prepare_receptor`; default `obabel`).

### Vina settings

- `--sf-name`: Vina scoring function (`vina`, `vinardo`, `ad4`; default `vina`).
- `--cpu`: Vina CPU count (`0` means all cores).
- `--seed`: Vina random seed (`0` means random seed).
- `--verbosity`: Vina verbosity (`0`, `1`, `2`).

## Pipeline Behavior

For each probe ligand, `cdock` performs:

1. Constrained 3D embedding onto the reference geometry.
2. Local optimization by Vina.
3. Optional OpenMM minimization with the receptor.
4. Final Vina rescoring and mapped heavy-atom RMSD calculation.
5. Pose write-out to SDF.

Each output molecule includes:

- `vina_score` (kcal/mol)
- `rmsd` (Angstrom)
- `ff_energy` (kJ/mol; only when EM runs)

CLI output reports:

- `vina_score`
- `rmsd`
- optional `ff_energy`
- `pose_sdf` path

## Usage Examples

Single probe ligand:

```bash
easybfe ligand cdock probes.sdf \
  -r ref.sdf \
  -p protein.pdb \
  -o pose.sdf
```

Batch mode (multiple probes in one file):

```bash
easybfe ligand cdock probes.sdf \
  -r ref.sdf \
  -p protein.pdb \
  -O poses/
```

Use explicit mapping:

```bash
easybfe ligand cdock probes.sdf \
  -r ref.sdf \
  -p protein.pdb \
  -O poses/ \
  --mapping-json mapping.json
```

Use explicit docking box and skip EM:

```bash
easybfe ligand cdock probes.sdf \
  -r ref.sdf \
  -p protein.pdbqt \
  -O poses/ \
  --box-center 10.0 12.5 8.0 \
  --box-size 18.0 16.0 14.0 \
  --no-em
```

## Validation and Common Failures

- Missing both `-o` and `-O`: command errors.
- Multiple probe records without `-O`: command errors.
- Only one of `--box-center` / `--box-size`: command errors.
- Invalid `--mapping-json` (not a JSON object): command errors.
- EM with non-`.pdb` protein input: runtime failure (`--no-em` avoids this).
- Empty probe/reference input files: command errors.

