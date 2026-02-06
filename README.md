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
cm-toy-addition
cm-make-figures
cm-opcode-test
cm-opcode16-test
cm-compound-opcode-test
cm-serial-adder-test
cm-kbit-memory-test
cm-kxor-visual
cm-rule110-gif
```

## Tests

```bash
pytest
```