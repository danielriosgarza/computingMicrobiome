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
