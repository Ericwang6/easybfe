# RBFE Setup Config Specification

The `easybfe rbfe setup` command reads a YAML or JSON config file validated as `AmberLigandRbfeConfig` (`easybfe.config.amber.rbfe`).

See [assets/config_rbfe_5ns.yaml](../assets/config_rbfe_5ns.yaml) for a complete working example.

## Top-Level Fields


| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `protein` | path | yes | Receptor PDB file |
| `ligandA` | path | no | Single-pair mode ligand A directory |
| `ligandB` | path | no | Single-pair mode ligand B directory |
| `ligand_list` | list of paths | no | Network mode ligand directories |
| `network` | object | no | Network-mode edge generation (`algorithm`, `options`) |
| `ligand_base` | path | no | Parent directory used to resolve `ligandA`/`ligandB`/`ligand_list` entries |
| `output_dir` | path | no | Output directory for single-pair mode |
| `output_base` | path | no | Output base for network mode, or parent for single-pair output naming |
| `atom_mapping` | object | no | Atom mapping settings (see below) |
| `solvent` | object | no | Solvent-leg FEP simulation config |
| `complex` | object | no | Complex-leg FEP simulation config |
| `gas` | object | no | Optional gas-phase leg |


### Input Modes

**Single pair** — set `ligandA`, `ligandB`, and `output_dir`:

```yaml
ligandA: ./ligands/ejm_44
ligandB: ./ligands/ejm_31
protein: ./protein.pdb
output_dir: ./rbfe/ejm_44~ejm_31
```

**Network** — set `ligand_list` with `network` and `output_base`:

```yaml
ligand_list:
  - "ejm_44"
  - "ejm_31"
  - "jmc_30"
ligand_base: ./ligands
network:
  algorithm: custom
  options:
    edges:
      - ["ejm_44", "ejm_31"]
      - ["ejm_44", "jmc_30"]
output_base: ./rbfe
protein: ./protein.pdb
```

## `network` — Network Config

`network` is used when `ligand_list` is set.

```yaml
network:
  algorithm: star
  options:
    center: ejm_44
```

See [rbfe_network_algorithms.md](rbfe_network_algorithms.md) for algorithm examples (`custom`, `star`, `wheel`, `bistar`, and optional OpenFE algorithms).

## `atom_mapping` — Atom Mapping Config

`AtomMappingConfig` controls how atoms in ligand A are mapped to ligand B for the alchemical transformation.


| Field       | Default     | Description                           |
| ----------- | ----------- | ------------------------------------- |
| `algorithm` | `kartograf` | Mapping backend: `lomap`, `kartograf` |
| `options`   | `{}`        | Algorithm-specific keyword arguments  |


### Kartograf (default)

```yaml
atom_mapping:
  algorithm: kartograf
  options:
    allow_map_hydrogen_to_non_hydrogen: true
    allow_hybridization_change: false
    allow_element_change: true
    atom_max_distance: 0.95
```

### LOMAP

```yaml
atom_mapping:
  algorithm: lomap
  options: {}
```

## `solvent` / `complex` — FEP Simulation Config

Both legs share the `AmberFepSimulationConfig` schema, identical to the ABFE legs. See [abfe_setup_spec.md](abfe_setup_spec.md) for the full field reference, including:

- System setup fields (`box_shape`, `buffer`, `neutralize`, `ionic_strength`, `do_hmr`, force fields, etc.)
- FEP-specific fields (`lambdas`, `num_lambdas`, `use_charge_change`, etc.)
- `workflow` stages and `cntrl` namelist

---

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

### Enable charge-change corrections

For transformations with net charge change, users can specify a special algorithm to alchemically modify a co-ion to a water. Enable this by setting `use_charge_change` to `true`. However, by default this parameter should be `false` because it is not stable yet. Only turn this on when the users ask you to do so.

```yaml
solvent:
  use_charge_change: true
complex:
  use_charge_change: true
```

### Switch atom mapping algorithm

```yaml
atom_mapping:
  algorithm: lomap
  options: {}
```

### Add a gas-phase leg

This is rarely used unless the user prompts to perform an energy decomposition analysis of the relative binding free energy or relative hydration free energy.

```yaml
gas:
  box_shape: cube
  buffer: 0.0
  gas_phase: true
  workflow:
    - type: em
      name: 01.em
      exec: pmemd
      use_remd: false
      use_mpi: false
      cntrl:
        maxcyc: 2000
    - type: prod
      name: 02.prod
      exec: pmemd
      use_remd: true
      use_mpi: true
      cntrl:
        nstlim: 125
        dt: 0.002
        numexchg: 5000
```

