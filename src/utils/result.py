from dataclasses import dataclass

import numpy as np


@dataclass
class IterativeResults:
    """A class to store the results of an iterative method.

    Attributes
    ----------
    solution: numpy.ndarray
        The estimated solution.
    residuals: list[float]
        The estimated residual on each iteration.
    errors: list[float]
        The true error on each iteration.
    time: float
        The execution time.
    iterations: int
        The number of iterations executed.
    exit_code: int
        The exit code from the iterative method.
    """

    solution: np.ndarray
    residuals: list[float]
    errors: list[float]
    time: float
    iterations: int
    exit_code: int
