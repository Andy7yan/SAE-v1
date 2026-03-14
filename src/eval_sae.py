from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import torch

import activation_store as activation_store_module
import sae as sae_module
from activation_store import FrozenActivationModel
from config import (
    DATA_CACHE_PATH,
    DOLMA_URL,
    HOOK_LAYER_INDEX,
    HTTP_TIMEOUT,
    MODEL_NAME,
    TEXT_BATCH_SIZE_PER_RANK,
    TEXT_PREFETCH_BACKEND,
    TEXT_PREFETCH_BATCHES,
    USER_AGENT,
)
from data import iter_text_batches
from sae import TinyJumpReLUSAE


def log(msg: str) -> None:
    print(f"[sae_eval] {msg}", flush=True)


activation_store_module.log0 = log


def ensure_local_dolma_shard_single_process() -> Path:
    if not DATA_CACHE_PATH.exists():
        DATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        log(f"Downloading Dolma shard to {DATA_CACHE_PATH} ...")
        request = urllib.request.Request(DOLMA_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, open(DATA_CACHE_PATH, "wb") as f:
            f.write(response.read())

    if not DATA_CACHE_PATH.exists():
        raise RuntimeError(f"Missing shard: {DATA_CACHE_PATH}")
    return DATA_CACHE_PATH


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["sae_state_dict"]
    d_in = int(state_dict["b_dec"].numel())
    d_latent = int(state_dict["b_enc"].numel())
    return {
        "checkpoint_path": checkpoint_path,
        "payload": payload,
        "d_in": d_in,
        "d_latent": d_latent,
        "hook_layer_index": int(payload.get("hook_layer_index", HOOK_LAYER_INDEX)),
        "model_name": str(payload.get("model_name", MODEL_NAME)),
        "activation_scale": float(payload.get("activation_scale", 1.0)),
    }


def build_sae_from_checkpoint(meta: dict, device: torch.device) -> TinyJumpReLUSAE:
    sae_module.LATENT_DIM = int(meta["d_latent"])
    sae_module.LATENT_FACTOR = None

    model = TinyJumpReLUSAE(d_in=int(meta["d_in"]))
    model.load_state_dict(meta["payload"]["sae_state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_checkpoint(
    meta: dict,
    activation_model: FrozenActivationModel,
    shard_path: Path,
    max_text_batches: int,
) -> tuple[Path, Path, Path]:
    checkpoint_path = Path(meta["checkpoint_path"])
    activation_scale = float(meta["activation_scale"])
    device = activation_model.device

    sae_model = build_sae_from_checkpoint(meta=meta, device=device)
    threshold = sae_model.get_threshold()
    d_latent = sae_model.d_latent
    d_in = sae_model.d_in

    reconstruction_rows: list[dict] = []
    sparsity_rows: list[dict] = []
    fired_counts = torch.zeros(d_latent, device=device, dtype=torch.int64)
    total_valid_tokens = 0
    used_text_batches = 0

    for texts in iter_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=0,
        world_size=1,
        prefetch_batches=TEXT_PREFETCH_BATCHES,
        prefetch_backend=TEXT_PREFETCH_BACKEND,
    ):
        if used_text_batches >= max_text_batches:
            break

        act, mask = activation_model.capture_text_batch(texts)
        x = act.reshape(-1, d_in)
        mask_flat = mask.reshape(-1)
        valid = x[mask_flat]

        if valid.numel() == 0:
            continue

        x_batch = (valid / activation_scale).to(device=device, dtype=torch.float32)
        x_hat, pre = sae_model(x_batch)

        reconstruction_loss = float(((x_hat - x_batch) ** 2).mean().item())
        fired = (pre > threshold).to(torch.int64)
        avg_l0 = float(fired.sum(dim=-1).float().mean().item())

        fired_counts += fired.sum(dim=0)
        n_valid_tokens = int(x_batch.shape[0])
        total_valid_tokens += n_valid_tokens

        reconstruction_rows.append(
            {
                "eval_batch_id": len(reconstruction_rows),
                "n_valid_tokens": n_valid_tokens,
                "reconstruction_loss": reconstruction_loss,
            }
        )
        sparsity_rows.append(
            {
                "eval_batch_id": len(sparsity_rows),
                "n_valid_tokens": n_valid_tokens,
                "avg_l0": avg_l0,
            }
        )

        used_text_batches += 1

    if total_valid_tokens == 0:
        raise RuntimeError(f"No valid tokens were collected for checkpoint: {checkpoint_path}")

    dead_feature_mask = fired_counts == 0
    dead_feature_count = int(dead_feature_mask.sum().item())
    total_feature_count = int(fired_counts.numel())
    dead_neuron_ratio = dead_feature_count / max(total_feature_count, 1)

    dead_rows = [
        {
            "checkpoint_name": checkpoint_path.name,
            "used_text_batches": used_text_batches,
            "total_valid_tokens": total_valid_tokens,
            "total_features": total_feature_count,
            "dead_features": dead_feature_count,
            "dead_neuron_ratio": dead_neuron_ratio,
        }
    ]

    reconstruction_path = checkpoint_path.parent / f"{checkpoint_path.stem}__reconstruction_loss.csv"
    sparsity_path = checkpoint_path.parent / f"{checkpoint_path.stem}__sparsity.csv"
    dead_path = checkpoint_path.parent / f"{checkpoint_path.stem}__dead_neuron_ratio.csv"

    write_csv(
        reconstruction_path,
        ["eval_batch_id", "n_valid_tokens", "reconstruction_loss"],
        reconstruction_rows,
    )
    write_csv(
        sparsity_path,
        ["eval_batch_id", "n_valid_tokens", "avg_l0"],
        sparsity_rows,
    )
    write_csv(
        dead_path,
        [
            "checkpoint_name",
            "used_text_batches",
            "total_valid_tokens",
            "total_features",
            "dead_features",
            "dead_neuron_ratio",
        ],
        dead_rows,
    )

    del sae_model
    torch.cuda.empty_cache()

    return reconstruction_path, sparsity_path, dead_path


def parse_args() -> argparse.Namespace:
    default_dir = Path("/srv/scratch/z5534565/sae-v1-res")
    default_checkpoints = [
        default_dir / "mar13-3000.pt",
        default_dir / "sae_step_005000.pt",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[str(path) for path in default_checkpoints],
        help="Checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--max-text-batches",
        type=int,
        default=64,
        help="Number of text batches to evaluate for each checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_metas = [load_checkpoint_metadata(path) for path in checkpoint_paths]

    hook_layer_values = {meta["hook_layer_index"] for meta in checkpoint_metas}
    if len(hook_layer_values) != 1:
        raise RuntimeError(
            "All checkpoints must use the same hook layer for this minimal evaluation script."
        )

    model_name_values = {meta["model_name"] for meta in checkpoint_metas}
    if len(model_name_values) != 1:
        raise RuntimeError(
            "All checkpoints must use the same frozen LLM for this minimal evaluation script."
        )

    device = torch.device("cuda:0")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    shard_path = ensure_local_dolma_shard_single_process()

    activation_model = FrozenActivationModel(
        device=device,
        model_dtype=model_dtype,
        hook_layer_index=int(checkpoint_metas[0]["hook_layer_index"]),
    )

    try:
        for meta in checkpoint_metas:
            checkpoint_path = Path(meta["checkpoint_path"])
            log(f"Evaluating: {checkpoint_path}")
            reconstruction_path, sparsity_path, dead_path = evaluate_checkpoint(
                meta=meta,
                activation_model=activation_model,
                shard_path=shard_path,
                max_text_batches=args.max_text_batches,
            )
            log(f"Wrote: {reconstruction_path}")
            log(f"Wrote: {sparsity_path}")
            log(f"Wrote: {dead_path}")
    finally:
        activation_model.close()


if __name__ == "__main__":
    main()
