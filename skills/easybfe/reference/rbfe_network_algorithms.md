# RBFE Network Algorithm Examples

Use these snippets in `network` when configuring RBFE with `ligand_list`.

## custom

```yaml
network:
  algorithm: custom
  options:
    edges:
      - ["ejm_44", "ejm_31"]
      - ["ejm_31", "jmc_30"]
      - ["ejm_44", "jmc_30"]
```

## star

```yaml
network:
  algorithm: star
  options:
    center: ejm_44
```

## wheel

```yaml
network:
  algorithm: wheel
  options:
    center: ejm_44
```

## bistar

```yaml
network:
  algorithm: bistar
  options:
    center1: ejm_44
    center2: ejm_31
```

## OpenFE-based (optional dependency)

```yaml
network:
  algorithm: minimal_spanning
  options: {}
```

Other available optional algorithms: `lomap`, `minimal_redundant`.
