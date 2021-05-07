import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.bottom"] = False
mpl.rcParams.update({'figure.autolayout': True})


def plot_settings():
    """
    Default settings for plots.
    """

    # grey grid with white lines
    plt.grid(True, "major", "both", color="w", linestyle="-", linewidth=2)
    plt.gca().patch.set_facecolor("0.8")
    plt.tight_layout()
