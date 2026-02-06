import numpy as np

from computingMicrobiome.models.k_xor import KXOR


def test_kxor_classifier():
    bits = 2
    model = KXOR(
        bits=bits,
        rule_number=110,
        width=100,
        boundary="periodic",
        recurrence=1,
        itr=1,
        d_period=10,
        seed=0,
    )

    model.fit()

    test_input = np.array([[1, 0]])
    predicted_output = model.predict(test_input)

    assert predicted_output.shape == (1,)
    assert predicted_output[0] == 1
