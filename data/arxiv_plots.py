import matplotlib.pyplot as plt
import numpy as np


def plot_ols_explanatory_power_fig():
    print("Plotting OLS explanatory power figure ...")

    metrics = ["Frequency", "Edge", "Superpixel", "Texture", "Entropy", "CNN"]

    mean_mcc = np.array([0.001, 0.005, 0.006, 0.006, 0.013, 0.023])
    best_mcc = np.array([0.010, 0.013, 0.020, 0.019, 0.011, 0.015])

    y = np.arange(len(metrics))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.barh(y - bar_height / 2, mean_mcc, height=bar_height, label="Mean MCC")
    ax.barh(y + bar_height / 2, best_mcc, height=bar_height, label="Best MCC")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 0.025)
    ax.set_xlabel(r"$R^2$ of single-metric OLS model")

    for i, value in enumerate(mean_mcc):
        ax.text(value + 0.0004, y[i] - bar_height / 2, f"{value:.3f}",
                va="center", ha="left", fontsize=9)

    for i, value in enumerate(best_mcc):
        ax.text(value + 0.0004, y[i] + bar_height / 2, f"{value:.3f}",
                va="center", ha="left", fontsize=9)

    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def get_latex_figure(print_output=True):
    print("Exploring OLS explanatory power figure ...\n\n")
    latex_code = r"""
        \begin{figure}[t]
        \centering
        \begin{tikzpicture}
        \begin{axis}[
            width=\linewidth,
            height=7cm,
            xbar,
            xmin=0, xmax=0.025,
            xlabel={$R^2$ of single-metric OLS model},
            symbolic y coords={Frequency,Edge,Superpixel,Texture,Entropy,CNN},
            ytick=data,
            legend style={at={(0.98,0.02)},anchor=south east,draw=none,fill=none},
            nodes near coords,
            nodes near coords align={horizontal},
            bar width=6pt,
            enlarge y limits=0.15
        ]
        \addplot coordinates {
            (0.001,Frequency)
            (0.005,Edge)
            (0.006,Superpixel)
            (0.006,Texture)
            (0.013,Entropy)
            (0.023,CNN)
        };
        \addplot coordinates {
            (0.010,Frequency)
            (0.013,Edge)
            (0.020,Superpixel)
            (0.019,Texture)
            (0.011,Entropy)
            (0.015,CNN)
        };
        \legend{Mean MCC, Best MCC}
        \end{axis}
        \end{tikzpicture}
        \caption{Explanatory power of individual similarity metrics in separate OLS models. CNN-based similarity provides the strongest single-metric signal for mean transfer performance, while superpixel- and texture-based similarity are strongest for best-case transfer performance. Overall, all effect sizes remain small.}
        \label{fig:single_metric_r2}
        \end{figure}
        """
    if print_output:
        print(latex_code)
    return latex_code

def main():
    plot_ols_explanatory_power_fig()
    get_latex_figure()

if __name__ == "__main__":
    main()