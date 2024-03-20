from usl_pipeline import adder


def test_addition():
    assert adder.add(1, 2) == 3
