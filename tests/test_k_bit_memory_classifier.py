import numpy as np

from computingMicrobiome.models.k_bit_memory import KBitMemory


def test_kbit_memory_classifier():
    bits = 4
    model = KBitMemory(
        bits=bits,
        rule_number=110,
        width=100,
        boundary="periodic",
        recurrence=2,
        itr=1,
        d_period=50,
        seed=0,
    )

    model.fit()

    test_input = np.array([[1, 0, 1, 0]])
    predicted_output = model.predict(test_input)

    assert predicted_output.shape == (1, bits)
