# computingMicrobiome

Elementary cellular automata reservoir computing benchmarks and experiments.

## Install

```bash
pip install -e .
```

Optional GIF support:

```bash
pip install -e .[gif]
```

## Quick usage

```python
from computingMicrobiome.models import KOpcodeLogic

model = KOpcodeLogic(
    rule_number=110,
    width=256,
    boundary="periodic",
    recurrence=8,
    itr=8,
    d_period=20,
).fit()

pred = model.predict([[0, 0, 0, 1, 1]])
```

## IBM universe composition

The individual-based microbiome (IBM) model uses a fixed **universe** of 50 species and 101 resources (100 nutrients + 1 toxic compound). You typically select a subset of species and get a minimal resource set for simulations or reservoir computing.

### Resource bands

- **High** (indices 0–9): top of the chain; main external feed.
- **Mid** (10–59): fed by high-eaters’ secretion and some feed.
- **Low** (60–99): fed by mid-eaters’ secretion and a little feed.
- **Toxin** (index 100): toxic compound; no uptake, absent from default media.

### Species groups (by index)

| Group         | Indices | Diet (primary) | Secretion              | Division cost |
|---------------|---------|----------------|------------------------|---------------|
| High-eaters   | 0–19    | High band      | 1–3 resources → mid/low | Highest (14–22) |
| Mid-eaters    | 20–39   | Mid band       | 1–2 → low              | Medium (10–16)  |
| Low-eaters    | 40–49   | Low band       | None                   | Lowest (6–12)   |

High-eaters always secrete into mid and/or low; low-eaters do not secrete (pure consumers). Division cost is structured so high-feeders pay more to divide than low-feeders.

### Toxic compound

- **Resource index 100**. The **last 3 species of each band** (17–19, 37–39, 47–49) secrete it.
- Each species has a **toxin tolerance**: if toxin concentration at a cell exceeds that value, the cell dies (applied in maintenance).
- Secretors are more tolerant (25–60); others have variable sensitivity (0–30).
- Toxin is **absent from the default media** (`feed_rate[100] = 0`). It can be added to the feed if desired.

### Default feed (media)

- **High band**: 3 resources at concentration 50–110 (main input).
- **Mid band**: 5 resources at 8–25.
- **Low band**: 5 resources at 0–5.
- **Toxin**: 0 (absent).

### Cross-feeding and curated sets

The chain high → mid → low gives cross-feeding: high-eaters feed mid-eaters via secretion; mid-eaters feed low-eaters. A curated 6-species set spanning the chain is available:

```python
from computingMicrobiome.ibm import CROSS_FEED_6_SPECIES, make_ibm_config_from_species, load_params

cfg = make_ibm_config_from_species(species_indices=CROSS_FEED_6_SPECIES, height=16, width_grid=32)
env, species = load_params(cfg)  # 6 species, minimal resource set including toxin
```

`CROSS_FEED_6_SPECIES` = `[0, 1, 20, 21, 40, 41]` (2 high + 2 mid + 2 low).

### Per-species energy

- **Energy capacity** defaults to 3× division cost (per species), so cells cannot accumulate unbounded energy.
- Config keys: `energy_capacity` (scalar or per-species), `toxin_tolerance` (per-species when using toxin).

## CLI entry points

After installation:

```bash
cm-make-figures
cm-opcode-test
cm-opcode16-test
cm-compound-opcode-test
cm-serial-adder-test
cm-kbit-memory-test
cm-kxor-visual
cm-rule110-gif
```

Toy-addition and readout-comparison experiments now live in top-level `experiments/`
and can be run with:

```bash
python -m experiments.run_toy_addition_experiment
python -m experiments.compare_readouts
python -m experiments.run_microbiome_host_evolution
```

Meta-evolutionary readout example (train, freeze, save, reload):

```bash
python examples/meta_evo_freeze_reload.py
```

## Tests

```bash
pytest
```

## Docs

Local preview:

```bash
pip install -e .[docs]
mkdocs serve
```

GitHub Pages is published via `.github/workflows/docs.yml` on pushes to `main`.
