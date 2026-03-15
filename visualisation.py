import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_eval_files(prefix: Path):
    recon_path = prefix.parent / f"{prefix.name}__reconstruction_loss.csv"
    sparsity_path = prefix.parent / f"{prefix.name}__sparsity.csv"
    dead_path = prefix.parent / f"{prefix.name}__dead_neuron_ratio.csv"

    if not recon_path.exists():
        raise FileNotFoundError(f"Missing file: {recon_path}")
    if not sparsity_path.exists():
        raise FileNotFoundError(f"Missing file: {sparsity_path}")
    if not dead_path.exists():
        raise FileNotFoundError(f"Missing file: {dead_path}")

    recon_df = pd.read_csv(recon_path)
    sparsity_df = pd.read_csv(sparsity_path)
    dead_df = pd.read_csv(dead_path)

    return recon_df, sparsity_df, dead_df, recon_path, sparsity_path, dead_path


def build_merged_df(recon_df: pd.DataFrame, sparsity_df: pd.DataFrame, total_features: int) -> pd.DataFrame:
    df = recon_df.merge(
        sparsity_df,
        on=["eval_batch_id", "n_valid_tokens"],
        how="inner",
        validate="one_to_one",
    )

    if total_features <= 0:
        raise ValueError("total_features must be positive.")

    df["active_fraction"] = df["avg_l0"] / total_features
    df["active_percent"] = df["active_fraction"] * 100.0

    df["reconstruction_loss_smooth"] = df["reconstruction_loss"].rolling(
        window=5,
        min_periods=1,
        center=True,
    ).mean()
    df["active_percent_smooth"] = df["active_percent"].rolling(
        window=5,
        min_periods=1,
        center=True,
    ).mean()

    return df


def save_summary_json(out_path: Path, df: pd.DataFrame, dead_df: pd.DataFrame):
    row = dead_df.iloc[0]

    summary = {
        "checkpoint_name": str(row["checkpoint_name"]),
        "used_text_batches": int(row["used_text_batches"]),
        "total_valid_tokens": int(row["total_valid_tokens"]),
        "total_features": int(row["total_features"]),
        "dead_features": int(row["dead_features"]),
        "dead_neuron_ratio": float(row["dead_neuron_ratio"]),
        "reconstruction_loss": {
            "mean": float(df["reconstruction_loss"].mean()),
            "std": float(df["reconstruction_loss"].std(ddof=1)),
            "min": float(df["reconstruction_loss"].min()),
            "max": float(df["reconstruction_loss"].max()),
        },
        "avg_l0": {
            "mean": float(df["avg_l0"].mean()),
            "std": float(df["avg_l0"].std(ddof=1)),
            "min": float(df["avg_l0"].min()),
            "max": float(df["avg_l0"].max()),
        },
        "active_percent": {
            "mean": float(df["active_percent"].mean()),
            "std": float(df["active_percent"].std(ddof=1)),
            "min": float(df["active_percent"].min()),
            "max": float(df["active_percent"].max()),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_reconstruction_vs_batch(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4.8))
    plt.scatter(df["eval_batch_id"], df["reconstruction_loss"], s=20)
    plt.plot(df["eval_batch_id"], df["reconstruction_loss_smooth"], linewidth=2)
    plt.xlabel("Eval batch id")
    plt.ylabel("Reconstruction loss")
    plt.title("SAE reconstruction loss by eval batch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_active_percent_vs_batch(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4.8))
    plt.scatter(df["eval_batch_id"], df["active_percent"], s=20)
    plt.plot(df["eval_batch_id"], df["active_percent_smooth"], linewidth=2)
    plt.xlabel("Eval batch id")
    plt.ylabel("Active features (%)")
    plt.title("SAE active fraction by eval batch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_recon_vs_active(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(df["active_percent"], df["reconstruction_loss"], s=28)
    plt.xlabel("Active features (%)")
    plt.ylabel("Reconstruction loss")
    plt.title("Reconstruction loss vs active fraction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_dead_ratio(dead_df: pd.DataFrame, out_path: Path):
    row = dead_df.iloc[0]
    dead_ratio_percent = float(row["dead_neuron_ratio"]) * 100.0
    alive_ratio_percent = 100.0 - dead_ratio_percent

    plt.figure(figsize=(7.2, 2.8))
    plt.barh(["Features"], [alive_ratio_percent], label="Alive")
    plt.barh(["Features"], [dead_ratio_percent], left=[alive_ratio_percent], label="Dead")
    plt.xlim(0, 100)
    plt.xlabel("Percentage of features")
    plt.title(
        f"Dead neuron ratio: {dead_ratio_percent:.2f}% "
        f"({int(row['dead_features'])}/{int(row['total_features'])})"
    )
    plt.legend()
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Checkpoint stem path, for example: /path/to/sae_step_005000",
    )
    args = parser.parse_args()

    prefix = Path(args.prefix)
    recon_df, sparsity_df, dead_df, recon_path, sparsity_path, dead_path = load_eval_files(prefix)

    total_features = int(dead_df.iloc[0]["total_features"])
    merged_df = build_merged_df(recon_df, sparsity_df, total_features=total_features)

    out_dir = prefix.parent
    stem = prefix.name

    plot_reconstruction_vs_batch(
        merged_df,
        out_dir / f"{stem}__plot_reconstruction_vs_batch.png",
    )
    plot_active_percent_vs_batch(
        merged_df,
        out_dir / f"{stem}__plot_active_percent_vs_batch.png",
    )
    plot_recon_vs_active(
        merged_df,
        out_dir / f"{stem}__plot_reconstruction_vs_active.png",
    )
    plot_dead_ratio(
        dead_df,
        out_dir / f"{stem}__plot_dead_neuron_ratio.png",
    )
    save_summary_json(
        out_dir / f"{stem}__plot_summary.json",
        merged_df,
        dead_df,
    )

    print("Loaded:")
    print(f"  {recon_path}")
    print(f"  {sparsity_path}")
    print(f"  {dead_path}")
    print("Wrote:")
    print(f"  {out_dir / f'{stem}__plot_reconstruction_vs_batch.png'}")
    print(f"  {out_dir / f'{stem}__plot_active_percent_vs_batch.png'}")
    print(f"  {out_dir / f'{stem}__plot_reconstruction_vs_active.png'}")
    print(f"  {out_dir / f'{stem}__plot_dead_neuron_ratio.png'}")
    print(f"  {out_dir / f'{stem}__plot_summary.json'}")


if __name__ == "__main__":
    main()