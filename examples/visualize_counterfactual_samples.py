from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def to_display_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).squeeze()
    # Handle tensors saved in [-1, 1] as well as already-normalized arrays.
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("/home/ruimin/experiment_data/exp_02_MNIST/counterfactual_sampling_class/samples.npz"),
        help="Path to samples.npz produced by sample_counterfactual.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./counterfactual_grid.png"),
        help="Path to save the visualization image",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to visualize",
    )
    args = parser.parse_args()

    samples = np.load(args.samples, allow_pickle=True)["arr_0"].item()
    originals = samples["original"][0]["image"].cpu().numpy()
    labels = samples["original"][0]["y"].cpu().numpy()
    counterfactuals = samples["counterfactual_sample"][0]

    num_examples = min(args.num_examples, len(originals), len(counterfactuals))
    fig, axes = plt.subplots(num_examples, 2, figsize=(4, 2 * num_examples))
    if num_examples == 1:
        axes = np.array([axes])

    for i in range(num_examples):
        axes[i, 0].imshow(to_display_image(originals[i]), cmap="gray")
        axes[i, 0].set_title(f"Original (y={labels[i]})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(to_display_image(counterfactuals[i]), cmap="gray")
        axes[i, 1].set_title("Counterfactual")
        axes[i, 1].axis("off")

    fig.suptitle("MNIST Counterfactual Samples", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
