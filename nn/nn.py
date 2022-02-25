# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Iterable, Callable, Optional
from numpy.typing import ArrayLike

VALID_ACTIVATIONS = ["sigmoid", "relu"]


def _collate(iterable: Iterable) -> Tuple[ArrayLike, ArrayLike]:
    """
    Transposes single element lists/tuples inside iterable along the reverse axis.
    So, instead of getting n lists of (x, y) pairs we return an n-long array of xs and an n-long vector/matrix of ys

    Args:
        iterable (Iterable): iterable of lists/tuples to be transposed

    Returns:
        Tuple[ArrayLike, ArrayLike]: transposed elements of iterable
    """

    xs, ys = list(zip(*iterable))

    xs = np.stack(xs)
    ys = np.stack(ys)
    assert len(xs) == len(
        ys
    ), "Collate fn did create equal length x and y matrices/vectors."

    assert len(xs) != 0, "Collating fn got an empty input"

    return xs, ys


def _batch_iterable(
    iterable: Iterable, n: int, collate_fn: Callable = _collate
) -> List[Tuple[ArrayLike, ArrayLike]]:
    """
    From an iterable, returns chunks of `iterable` that are n or less sized. If we reach the end of
    the iterable and the remaining length data are of size less than n, we return that size list.

    Args:
        iterable (Iterable): any iterable
        n (int): max size of returned batches

    Yields:
        List[List[ArrayLike, ArrayLike]]: within this module, used to return a list of lists of arrays that are (x, y) pairs

    """

    batch_count = 0
    batch = []

    for item in iterable:
        batch.append(item)
        batch_count += 1

        if batch_count == n:
            yield collate_fn(batch)  # will receive a list of lists and transpose it

            # reset batch counters
            batch_count = 0
            batch = []

    if batch:  # in case we have an uneven size batch
        yield collate_fn(batch)


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, int]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str,
        activations: Optional[List[str]] = None,
    ):

        # inputhandling for activations argument
        if activations is None:
            self._activations = [None] * len(nn_arch)
        else:
            if isinstance(activations, (list, tuple)) is False:
                raise TypeError("Activations must be a list or tuple.")

            if len(activations) != len(nn_arch):
                raise ValueError("Activations must be same length as architecture.")

            self._activations = []
            for activation in activations:
                if activation is None:
                    self._activations.append(None)
                elif type(activation) == str:
                    activation_lower = activation.lower()

                    if activation_lower not in VALID_ACTIVATIONS:
                        raise NotImplementedError(
                            f"Activation {activation} not implemented."
                        )

                    self._activations.append(activation_lower)
                else:
                    raise TypeError("Activations must either be none or string.")

        # input handling for loss fn
        try:
            loss_lower = loss_function.lower()
            self._loss_func = loss_lower
        except:
            raise TypeError("Argument loss_functions must be a string.")

        if loss_lower not in ["mse", "ce"]:
            raise ValueError('Loss must be one of "mse" or "ce".')

        # Saving architecture
        self.arch = nn_arch
        self.n_layers = len(nn_arch)  # just to keep track of number of layers
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_lower
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

        for idx, activation in enumerate(self._activations):
            self._param_dict[f"f{idx+1}"] = activation

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer["input_dim"]
            output_dim = layer["output_dim"]
            # initializing weight matrices
            param_dict["W" + str(layer_idx)] = (
                np.random.randn(output_dim, input_dim) * 0.1
            )
            # initializing bias matrices
            param_dict["b" + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(
        self, W_curr: ArrayLike, b_curr: ArrayLike, A_prev: ArrayLike, activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        z = A_prev.dot(W_curr.T)
        z += b_curr.T
        if activation:
            if activation == "sigmoid":
                a = self._sigmoid(z)
            elif activation == "relu":
                a = self._relu(z)
            else:
                raise ValueError(
                    f'Activation {activation} was passed into _single_forward but was not one of "sigmoid", "relu" and could not be mapped to a function.'
                )
        else:
            a = z.copy()

        return a, z

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        _a = X
        backprop_dict = {}

        for layeridx in range(self.n_layers):
            w_key = f"W{layeridx+1}"
            b_key = f"b{layeridx+1}"
            activation_key = f"f{layeridx+1}"

            W = self._param_dict[w_key]
            b = self._param_dict[b_key]
            activation = self._param_dict[activation_key]

            a, z = self._single_forward(W, b, _a, activation)
            backprop_dict[f"a{layeridx+1}"] = a
            backprop_dict[f"z{layeridx+1}"] = z
            _a = a

        return a, backprop_dict

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation: str,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:

        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation == "sigmoid":
            dA_dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation == "relu":
            dA_dZ = self._relu_backprop(dA_curr, Z_curr)
        else:
            dA_dZ = A_prev

        dA_prev = dA_dZ * dA_curr  # dLoss / dActivation * dActivation / dZ
        print("wcurr", W_curr.shape, "daprev", dA_prev.shape)
        dW_curr = dA_prev.copy()
        print(dW_curr.shape)
        print()
        db_curr = dA_prev.copy()
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        loss, _dA = self._get_loss(
            y_hat, y, self._loss_func, return_derivative=True
        )  # derivative of loss wrt activations

        print("loss, da", loss.shape, _dA.shape)

        grad_dict = {}

        for layeridx in reversed(range(self.n_layers)):
            w_key = f"W{layeridx+1}"
            b_key = f"b{layeridx+1}"
            activation_key = f"f{layeridx+1}"
            a_key = f"a{layeridx+1}"
            z_key = f"z{layeridx+1}"

            W = self._param_dict[w_key]
            b = self._param_dict[b_key]

            a = cache[a_key]
            z = cache[z_key]

            activation = self._param_dict[activation_key]

            _dA, dW, db = self._single_backprop(W, b, z, a, _dA, activation)
            grad_dict[w_key] = dW
            grad_dict[b_key] = db

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for layeridx in range(self.n_layers):
            w_key = f"W{layeridx+1}"
            b_key = f"b{layeridx+1}"

    #            self._param_dict[w_key] += self._lr * self.

    def _get_loss(
        self, y_hat: ArrayLike, y: ArrayLike, loss: str, return_derivative=True
    ):

        losses = []
        assert self._loss_func in [
            "mse",
            "ce",
        ], 'Loss function was unmapped from "mse" or "ce"; replace with one of those to continue.'

        if self._loss_func == "mse":
            loss = self._mean_squared_error(y, y_hat)
        elif self._loss_func == "ce":
            loss = self._binary_cross_entropy(y, y_hat)

        losses.append(loss)

        losses = np.mean(losses)

        if return_derivative:
            if self._loss_func == "mse":
                derivative = self._mean_square_error_backprop(y, y_hat)
            if self._loss_func == "ce":
                derivative = self._binary_cross_entropy_backprop(y, y_hat)
            return losses, derivative

        return losses

    def _eval_loader(self, X: ArrayLike, y: ArrayLike, train: bool = True):

        losses = []
        for x, y in _batch_iterable(zip(X, y), self._batch_size):
            x, back_dict = self.forward(x)

            if train:
                grad_dict = self.backprop(y, x, back_dict)
                self._update_params({})

            else:
                losses = self._get_loss(x, y, self._loss_func)

        return losses

    def fit(
        self, X_train: ArrayLike, y_train: ArrayLike, X_val: ArrayLike, y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        train_losses = self._eval_loader(X_train, y_train)
        test_losses = self._eval_loader(X_val, y_val)
        return train_losses, test_losses

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        pass

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function. See https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        z_sigmoid = np.zeros_like(Z)
        positive_mask = Z > 0
        negative_mask = ~positive_mask

        exp_z = np.exp(Z[negative_mask])
        z_sigmoid[negative_mask] = exp_z / (exp_z + 1)
        z_sigmoid[positive_mask] = 1 / (1 + np.exp(-Z[positive_mask]))

        return z_sigmoid

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.clip(Z, 0, None)  # thresholds at 0

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sigmoid = self._sigmoid(dA)
        activation_derivative = sigmoid * (1 - sigmoid)

        print()
        print("actn deriv", activation_derivative.shape)
        print("z", Z.shape, "a", dA.shape)

        # z_derivative = np.full(Z.shape, activation_derivative)
        return activation_derivative

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        activation_derivative = np.where(dA > 1, 1, 0)
        z_derivative = np.full(dA.shape, activation_derivative)
        z_derivative *= z_derivative
        return z_derivative

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        exp_sum = np.exp(y_hat)
        softmax_probs = exp_sum / exp_sum.sum()

        loss = y * np.log(softmax_probs)
        return -loss

    def _binary_cross_entropy_backprop(
        self, y: ArrayLike, y_hat: ArrayLike
    ) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        exp_sum = np.exp(y_hat)
        softmax_probs = exp_sum / exp_sum.sum()
        return softmax_probs - y

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.power(y - y_hat, 2)
        return loss.mean()

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return 2 * np.mean(y - y_hat)

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass
