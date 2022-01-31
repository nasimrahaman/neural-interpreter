import pytest


def test_ni():
    import torch
    from neural_interpreters.core.interface import NeuralInterpreter
    from neural_interpreters.utils import read_yaml
    from pathlib import Path

    config = read_yaml(
        (Path(__file__) / ".." / ".." / "presets" / "digits.yml").resolve().absolute()
    )

    ni = NeuralInterpreter(**config.model.kwargs)

    x = torch.randn(1, 3, 32, 32)
    output = ni(x)

    assert list(output.predictions["mnist"].shape) == [1, 10]
    assert list(output.predictions["mnistm"].shape) == [1, 10]
    assert list(output.predictions["svhn"].shape) == [1, 10]


if __name__ == "__main__":
    pytest.main()
