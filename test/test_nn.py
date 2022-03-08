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
    da = -np.load("test/da.npy")  # accidentally saved these as negatives
    db = -np.load("test/db.npy")
    dw = -np.load("test/dw.npy")

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


def test_predict(network, data):
    output = np.load("./test/outputs.npy")
    test_output = network.predict(data)

    assert np.allclose(
        output, test_output
    ), "Output and test output for predict function not equal"


def test_binary_cross_entropy(network):
    np.random.seed(0)
    logits = np.clip(np.random.randn(10, 2), 1e-5, 1 - 1e-5)
    target = np.zeros((10, 2))
    target[:5, 0] = 1
    target[5:, 1] = 1

    loss = network._binary_cross_entropy(target, logits)
    assert np.isclose(
        loss, 5.2437839
    ), "Binary cross entropy loss computation not close to manually calculated value"


def test_binary_cross_entropy_backprop(network):
    np.random.seed(0)
    logits = np.clip(np.random.randn(10, 2), 1e-5, 1 - 1e-5)
    target = np.zeros((10, 2))
    target[:5, 0] = 1
    target[5:, 1] = 1

    grad = network._binary_cross_entropy_backprop(target, logits)
    assert np.isclose(
        grad.mean(), -250.007
    ), "Binary cross entropy backward computation not close to manually calculated value"


def test_mean_squared_error(network):
    np.random.seed(0)
    inputs = np.random.randn(10, 2)
    targets = np.random.randn(10, 2)

    loss = network._mean_squared_error(targets, inputs)
    assert np.isclose(
        loss, 2.2209999
    ), "Mean squared error computation not close to manually calculated value"


def test_mean_squared_error_backprop(network):
    np.random.seed(0)
    inputs = np.random.randn(10, 2)
    targets = np.random.randn(10, 2)

    grad = network._mean_squared_error_backprop(inputs, targets)
    assert np.isclose(grad.mean(), -0.051358)


def test_one_hot_encode():
    seqs = ["AGA", "ACTG"]

    encodings_test = preprocess.one_hot_encode_seqs(seqs)

    encodings = [
        np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
    ]
    assert np.allclose(encodings[0], encodings_test[0]) and np.allclose(
        encodings[1], encodings_test[1]
    ), "Encodings incorrect"

    assert len(encodings) == len(encodings_test), "Returned wrong number of encodings"


def test_sample_seqs():
    data = np.random.randn(10, 1).tolist()
    labels = [0] * 10
    labels[0] = 1

    undersampled_data_pt = data[0]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(data, labels)
    assert (
        sampled_labels.count(0) == 9 and sampled_labels.count(1) == 9
    ), "Class counts are not correct"
    assert (
        sampled_seqs.count(undersampled_data_pt) == 9
    ), "Undersampled data point was not included to correct count"
