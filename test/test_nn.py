# BMI 203 Project 7: Neural Network

# Import necessary dependencies here

import numpy as np
from nn import nn, preprocess
import pickle
import pytest


@pytest.fixture
def network():
    arch = [{"input_dim": 5, "output_dim": 3}, {"input_dim": 3, "output_dim": 1}]
    network = nn.NeuralNetwork(
        arch, 1e-2, 0, 200, 10, "mse", activations=["sigmoid", "sigmoid"]
    )
    return network


@pytest.fixture
def data():
    return np.load("./test/input-forward.npy")


@pytest.fixture
def forward_dict():
    with open("./test/forward-dict.pkl", "rb") as f:
        forward_dict = pickle.load(f)
    return forward_dict


def test_forward(network, data, forward_dict):
    output = np.load("./test/outputs.npy")

    test_output, test_dict = network.forward(data)

    assert np.allclose(test_output, output), "Forward outputs do not match"
    for key in test_dict.keys():
        assert np.allclose(
            test_dict[key], forward_dict[key]
        ), f"Forward layer values for layer {key} do not match"


def test_single_forward(network, data, forward_dict):
    a, z = network._single_forward(
        network._param_dict["W1"], network._param_dict["b1"], data, "sigmoid"
    )

    assert np.allclose(a, forward_dict["a1"]), "A1 for first layer not similar"
    assert np.allclose(z, forward_dict["z1"]), "Z1 for first layer not similar"


def test_single_backprop(network, data):
    da = np.load("test/da.npy")
    db = np.load("test/db.npy")
    dw = np.load("test/dw.npy")

    test_output, cache = network.forward(data)

    loss, dL_dA = network._get_loss(
        test_output, np.ones(len(data)), "mse", return_derivative=True
    )

    w = network._param_dict["W2"]
    b = network._param_dict["b2"]
    a = cache["a1"]
    z = cache["z2"]

    _dA, _dW, _db = network._single_backprop(w, b, z, a, dL_dA, "sigmoid")

    assert np.allclose(da, _dA)
    assert np.allclose(db, _db)
    assert np.allclose(dw, _dW)


def test_predict():
    pass


def test_binary_cross_entropy(network):
    np.random.seed(0)
    logits = np.random.randn(10, 2)
    target = np.ones(10).reshape(-1, 1)

    loss = network._binary_cross_entropy(logits, target)
    assert np.isclose(loss, 1.3109413466424518)


def test_binary_cross_entropy_backprop(network):
    np.random.seed(0)
    logits = np.random.randn(10, 2)
    target = np.ones(10).reshape(-1, 1)

    grad = network._binary_cross_entropy_backprop(target, logits).mean()

    assert np.isclose(grad, -0.95)


def test_mean_squared_error(network):
    np.random.seed(0)
    inputs = np.random.randn(10, 1)


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
