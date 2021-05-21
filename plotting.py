import seaborn as sns
import matplotlib.pyplot as plt

# Using seaborn's style
plt.style.use('seaborn-white')

WIDTH = 345
GR = (5**.5 - 1) / 2
FORMAT = "pdf"

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "font.size": 18,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
}

plt.rcParams.update(tex_fonts)
plt.rcParams.update({"legend.handlelength": 1.5})
plt.rcParams.update({
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    })

def savefig(filename):
    plt.savefig(
        filename + "." + FORMAT, format=FORMAT, dpi=1200, bbox_inches="tight")


def line_plot(
        Y, X, xlabel=None, ylabel=None, ymax=None, ymin=None,
        xmax=None, xmin=None, filename=None, legend=None, errors=None,
        xlog=False, ylog=False, size=None, marker="s", color="k", linestyle=None):
    plt.clf()
    if legend is None:
        legend = [None] * Y.shape[0]

    if size is not None:
        plt.figure(figsize=size)

    if isinstance(color, str):
        color = [color] * Y.shape[0]
    if isinstance(marker, str):
        marker = [marker] * Y.shape[0]
    if linestyle is None:
        linestyle = ["-"] * Y.shape[0]

    for n in range(Y.shape[0]):
        x = X[n, :] if X.ndim == 2 else X
        plt.plot(x, Y[n, :], label=legend[n],
                marker=marker[n], markersize=6,
                linestyle=linestyle[n],
                color=color[n])
        if errors is not None:
            plt.fill_between(
                x, Y[n, :] - errors[n, :], Y[n, :] + errors[n, :],
                alpha=0.1, color=color[n])

    if ymax is not None:
        plt.ylim(top=ymax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if xmax is not None:
        plt.xlim(right=xmax)
    if xmin is not None:
        plt.xlim(left=xmin)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend[0] is not None:
        plt.legend(loc="upper left")

    axes = plt.gca()
    if xlog:
        axes.semilogx(10.)
    if ylog:
        axes.semilogy(10.)

    if filename is not None:
        savefig(filename)
