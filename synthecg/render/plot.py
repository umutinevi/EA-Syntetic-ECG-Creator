import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from synthecg.config import LEAD_NAMES, RenderConfig


def plot_ecg_layout(record, output_path: str, config: RenderConfig | None = None) -> None:
    """Render a 3x4 + rhythm strip ECG layout to a PNG file."""
    config = config or RenderConfig()
    signals = record.p_signal.T
    fs = record.fs

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.set_xlim(0, 10)
    baselines = [7.5, 5.0, 2.5, 0]
    ax.set_ylim(-1.5, 9.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.grid(which="major", color="#ff9999", linestyle="-", linewidth=1.2)
    ax.grid(which="minor", color="#ffcccc", linestyle="-", linewidth=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)

    time = np.arange(signals.shape[1]) / fs
    col_bounds = [(0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]

    for row in range(3):
        for col in range(4):
            lead_idx = col * 3 + row
            if lead_idx < 12:
                t_start, t_end = col_bounds[col]
                idx_start = int(t_start * fs)
                idx_end = int(t_end * fs)

                t_segment = time[idx_start:idx_end]
                sig_segment = signals[lead_idx, idx_start:idx_end]
                baseline = baselines[row]

                ax.plot(t_segment, sig_segment + baseline, color="#000033", linewidth=1.0)
                ax.text(
                    t_start + 0.05,
                    baseline + 1.0,
                    LEAD_NAMES[lead_idx],
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    backgroundcolor="white",
                )

    rhythm_sig = signals[1, : int(10 * fs)]
    ax.plot(time[: int(10 * fs)], rhythm_sig + baselines[3], color="#000033", linewidth=1.0)
    ax.text(
        0.05,
        baselines[3] + 1.0,
        "II",
        fontsize=12,
        fontweight="bold",
        color="black",
        backgroundcolor="white",
    )

    ax.text(
        8.2,
        -1.2,
        f"{config.speed_mm_s} mm/s   {config.gain_mm_mv} mm/mV",
        fontsize=10,
        fontweight="bold",
        color="black",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
