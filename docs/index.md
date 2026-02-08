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

## Quick start

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

## Build docs locally

```bash
pip install -e .[docs]
mkdocs serve
```
