"""Static figures (matplotlib). Fixed method -> colour/marker mapping; colour never carries identity alone.
Palette: validated categorical order (blue, orange, aqua, yellow, magenta); every series also has its own
marker, a legend, and the numbers are written to CSV next to each figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4",
         "exact": "#222222", "SA": "#7a7a7a", "tabu": "#7a7a7a", "greedy": "#7a7a7a"}
MARKER = {"M1": "o", "M2": "v", "M3": "s", "M4": "^", "M5": "D", "exact": "X", "SA": "P", "tabu": "*", "greedy": "h"}
LABEL = {"M1": "M1 MPSX-QAOA (proposed, multi-angle)", "M2": "M2 MPSX-QAOA (single-angle)", "M3": "M3 QAOA + slack penalty",
         "M4": "M4 QAOA + unbalanced penalty", "M5": "M5 multi-angle particle-conserving (MA-PC)"}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
