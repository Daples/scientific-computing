import os

import matplotlib.pyplot as plt


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _levels: int
        The numer of levels for a countour plot.
    _folder: str
        The folder to store the output figures.
    """

    _folder: str = "figs"

    @staticmethod
    def clear() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

    plt.rc("text", usetex=True)
    plt.rcParams.update({"font.size": 18})

    @classmethod
    def _add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.

        Parameters
        ----------
        path: str
            A path in string.

        Returns
        -------
        str
            The path with the added folder.
        """

        return os.path.join(cls._folder, path)

    @classmethod
    def get_plot(
        cls,
        x: list[float],
        y: list[float],
        path: str,
        xlabel: str = "$\log_2N$",
        ylabel: str = "Quantity",
    ) -> None:
        """It plots the best-so-far misclassification error between iterations.

        Parameters
        ----------

        """

        cls.clear()
        cls.__setup_config__()

        plt.plot(x, y, "k-o", markersize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(cls._add_folder(path), bbox_inches="tight")
