# ABFE Setup Config Specification

The `easybfe abfe setup` command reads a YAML or JSON config file validated as `AmberAbfeConfig` (`easybfe.config.amber.abfe`).

See [assets/config_abfe_5ns.yaml](../assets/config_abfe_5ns.yaml) for a complete working example.

## Top-Level Fields


| Field          | Type          | Required | Description                                                                            |
| -------------- | ------------- | -------- | -------------------------------------------------------------------------------------- |
| `protein`      | path          | yes      | Receptor PDB file                                                                      |
| `ligand`       | path          | no       | Single ligand directory (mutually exclusive with `ligand_batch`)                       |
| `ligand_batch` | list of paths | no       | Multiple ligand directories                                                            |
| `ligand_base`  | path          | no       | Parent directory; ligand paths are resolved relative to this                           |
| `output_dir`   | path          | no       | Output directory (single-ligand mode)                                                  |
| `output_base`  | path          | no       | Base output directory (batch mode; each ligand writes to `output_base/{ligand_name}/`) |
| `boresch`      | object        | yes      | Boresch restraint settings (see below)                                                 |
| `solvent`      | object        | yes      | Solvent-leg FEP simulation config                                                      |
| `complex`      | object        | yes      | Complex-leg FEP simulation config                                                      |
| `restraint`    | object        | yes      | Restraint-leg FEP simulation config                                                    |


### Input Modes

**Single ligand** — set `ligand` (directory) and `output_dir`:

```yaml
ligand: ./ligands/jmc_23
protein: ./protein.pdb
output_dir: ./abfe/jmc_23
```

**Batch** — set `ligand_batch` (list) with `ligand_base` and `output_base`:

```yaml
ligand_base: ./ligands
ligand_batch:
  - jmc_23
  - ejm_31
protein: ./protein.pdb
output_base: ./abfe
```

## `boresch` — Boresch Restraint Settings

`BoreschRestraintGeneratorConfig`. Use `{}` for defaults.


| Field       | Default                    | Description                                                          |
| ----------- | -------------------------- | -------------------------------------------------------------------- |
| `algorithm` | `rxrx`                     | Restraint placement algorithm                                        |
| `rst_wts`   | `[10, 10, 10, 10, 10, 10]` | Force constants for the 6 Boresch DOFs (bond, 2 angles, 3 dihedrals) |
| `options`   | `{}`                       | Algorithm-specific keyword arguments                                 |


```yaml
boresch: {}
```

## `solvent` / `complex` / `restraint` — FEP Simulation Config

All three legs share the `AmberFepSimulationConfig` schema (inherits `SetupConfig`). ABFE has three legs; the `restraint` leg handles the Boresch restraint window.

### System Setup Fields (from `SetupConfig`)


| Field            | Default    | Description                                                       |
| ---------------- | ---------- | ----------------------------------------------------------------- |
| `box_shape`      | `cube`     | Periodic box: `cube`, `dodecahedron`, or `octahedron`             |
| `buffer`         | `20.0`     | Solvent padding in Angstrom                                       |
| `neutralize`     | `true`     | Add counter-ions                                                  |
| `ionic_strength` | `0.15`     | NaCl concentration in mol/L                                       |
| `do_hmr`         | `true`     | Hydrogen mass repartitioning                                      |
| `hydrogen_mass`  | `3.024`    | Target H mass (amu) when HMR is on                                |
| `water_model`    | `tip3p`    | Water model (`tip3p`, `spce`, `tip4pew`, `tip5p`, `swm4ndp`)      |
| `protein_ff`     | `[ff14sb]` | Protein force field XMLs (shorthand `ff14sb` → `amber14-all.xml`) |
| `water_ff`       | `[tip3p]`  | Water force field XMLs (shorthand `tip3p` → `amber14/tip3p.xml`)  |
| `extra_ff`       | `[]`       | Additional force field XMLs                                       |
| `num_procs`      | `-1`       | Parallelism hint; `<=0` means auto (set to `num_lambdas`)         |
| `basename`       | `system`   | File basename for topology/coordinate files                       |


### FEP-Specific Fields


| Field                             | Default     | Description                                                                 |
| --------------------------------- | ----------- | --------------------------------------------------------------------------- |
| `lambdas`                         | `null`      | Explicit lambda schedule (list of floats 0.0–1.0). Overrides `num_lambdas`. |
| `num_lambdas`                     | `16`        | Number of evenly spaced lambdas (used only when `lambdas` is null)          |
| `use_charge_change`               | `true`      | Enable charge-changing FEP corrections                                      |
| `charge_change_method`            | `dummy_ion` | Method: `dummy_ion` or `coalchem_water`                                     |
| `use_settle_for_alchemical_water` | `true`      | SETTLE constraints on alchemical water                                      |
| `add_restraint_for_alchem_water`  | `true`      | Restraints on alchemical water                                              |
| `reduce_storage`                  | `true`      | Reduce disk usage for trajectories                                          |


### `workflow` — MD Stages

Ordered list of `AmberStepConfig` objects. Each stage type auto-populates sensible Amber `cntrl` defaults; user-provided values override them.


| Field      | Default      | Description                                                            |
| ---------- | ------------ | ---------------------------------------------------------------------- |
| `type`     | `prod`       | Stage type: `em`, `heat`, `pres`, `prod`, `prod_nvt`                   |
| `name`     | —            | Subdirectory name (e.g. `01.em`, `05.prod`)                            |
| `exec`     | `pmemd.cuda` | Amber executable: `pmemd`, `pmemd.cuda`, `pmemd.MPI`, `pmemd.cuda.MPI` |
| `use_remd` | `true`       | Hamiltonian REMD across lambda windows                                 |
| `use_mpi`  | `true`       | MPI parallelization. Must be `false` for `type: em`.                   |
| `cntrl`    | —            | Amber `&cntrl` namelist settings (see below)                           |


### Key `cntrl` Fields


| Field      | Default  | Description                                             |
| ---------- | -------- | ------------------------------------------------------- |
| `nstlim`   | `10000`  | Number of MD steps                                      |
| `dt`       | `0.001`  | Timestep in ps                                          |
| `ofreq`    | `1000`   | Output frequency (sets `ntwr`, `ntwx` if not specified) |
| `maxcyc`   | `2000`   | Max minimization cycles (for `type: em`)                |
| `numexchg` | `0`      | Number of REMD exchange attempts (for `use_remd: true`) |
| `temp0`    | `298.15` | Target temperature (K)                                  |
| `tempi`    | `0.0`    | Initial temperature (K)                                 |
| `cut`      | `10.0`   | Non-bonded cutoff (Angstrom)                            |
| `ntp`      | `1`      | Pressure coupling (0=none, 1=isotropic)                 |
| `ntb`      | `2`      | Boundary (1=const volume, 2=const pressure)             |
| `barostat` | `2`      | Barostat (1=Berendsen, 2=Monte Carlo)                   |
| `gamma_ln` | `2.0`    | Langevin collision frequency (ps^-1)                    |


## Typical 5-Stage Workflow

Note: you should always use `pmemd.cuda` as exec unless the user explicitly ask to to *run simulation* on CPU (which is rarely asked). The user may ask you to run the setup on CPU, but in this case should still use `pmemd.cuda` because the users may run simulations on GPU afterwards.

```yaml
workflow:
  - type: em            # 1) Energy minimization
    name: 01.em
    exec: pmemd.cuda
    use_remd: false
    use_mpi: false
    cntrl:
      maxcyc: 2000
      ofreq: 100

  - type: heat          # 2) Heating 0 → 298.15 K
    name: 02.heat
    exec: pmemd.cuda
    use_remd: false
    use_mpi: true
    cntrl:
      nstlim: 12500
      dt: 0.002         # 25 ps total
      ofreq: 1250

  - type: pres          # 3) Pressure equilibration
    name: 03.pres
    exec: pmemd.cuda
    use_remd: false
    use_mpi: true
    cntrl:
      nstlim: 12500
      dt: 0.002         # 25 ps total
      ofreq: 1250

  - type: prod          # 4) Pre-production (no REMD, decorrelation)
    name: 04.pre_prod
    exec: pmemd.cuda
    use_remd: false
    use_mpi: true
    cntrl:
      nstlim: 125000
      dt: 0.004         # 500 ps total
      ofreq: 12500

  - type: prod          # 5) Production with H-REMD
    name: 05.prod
    exec: pmemd.cuda
    use_remd: true
    use_mpi: true
    cntrl:
      nstlim: 125       # steps between exchanges
      dt: 0.004
      ofreq: 12500
      numexchg: 10000   # 125 * 0.004 * 10000 = 5000 ps = 5 ns
```

---

## Common Modifications

### Change production simulation time

**Simulation time formulas:**

- Standard MD: `total_time = nstlim * dt`
- REMD production: `total_time = nstlim * dt * numexchg`

Adjust `numexchg` in the final `prod` stage. With `nstlim: 125` and `dt: 0.004`:

```yaml
# 2 ns production
cntrl:
  nstlim: 125
  dt: 0.004
  numexchg: 4000    # 125 * 0.004 * 4000 = 2000 ps

# 10 ns production
cntrl:
  nstlim: 125
  dt: 0.004
  numexchg: 20000   # 125 * 0.004 * 20000 = 10000 ps
```

### Change lambda schedule

#### **Default**

The default is 16 evenly distributed lambdas:

```yaml
lambdas: 
  - 0.0
  - 0.06666666666666667
  - 0.13333333333333333
  - 0.2
  - 0.26666666666666666
  - 0.3333333333333333
  - 0.4
  - 0.4666666666666667
  - 0.5333333333333333
  - 0.6
  - 0.6666666666666666
  - 0.7333333333333333
  - 0.8
  - 0.8666666666666667
  - 0.9333333333333333
  - 1.0
num_lambdas: 16
```

**More lambdas (24 evenly-distributed):**

```yaml
lambdas: [
      0.0        , 0.04347826, 0.08695652, 0.13043478, 0.17391304,
      0.2173913 , 0.26086957, 0.30434783, 0.34782609, 0.39130435,
      0.43478261, 0.47826087, 0.52173913, 0.56521739, 0.60869565,
      0.65217391, 0.69565217, 0.73913043, 0.7826087 , 0.82608696,
      0.86956522, 0.91304348, 0.95652174, 1.0
]
num_lambdas: 24
```

Note that the `num_lambdas` field will be override by `lambdas`.

### Use a different box shape for the complex

Dodecahedron reduces the number of water molecules compared to a cube:

```yaml
complex:
  box_shape: dodecahedron
  buffer: 15.0

solvent:
  box_shape: cube
  buffer: 20.0
```

### Shorter restraint-leg production

The restraint leg often converges faster. Use a smaller `numexchg`:

```yaml
restraint:
  workflow:
    # ... (same em/heat/pres/pre_prod stages) ...
    - type: prod
      name: 05.prod
      exec: pmemd.cuda
      use_remd: true
      use_mpi: true
      cntrl:
        nstlim: 125
        dt: 0.004
        numexchg: 5000    # 2.5 ns instead of 5 ns
```

### Enable charge-change corrections

For transformations with net charge change, users can specify a special algorithm to alchemically modify a co-ion to a water. Enable this by setting `use_charge_change` to `true`. However, by default this parameter should be `false` because it is not stable yet. Only turn this on when the users ask you to do so.

```yaml
solvent:
  use_charge_change: true
complex:
  use_charge_change: true
```

