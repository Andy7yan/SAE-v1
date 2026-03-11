import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b-it"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    # Option A: read from environment variable
    hf_token = os.environ.get("HF_TOKEN")

    # Option B: if you already ran `hf auth login`, you may leave hf_token as None.
    # The library can use the stored token automatically.
    #
    # Do NOT hardcode your real token into source files that may be committed to git.

    device = get_device()
    model_dtype = get_model_dtype(device)

    print(f"device={device}")
    print(f"model_dtype={model_dtype}")
    print(f"HF_TOKEN_present={hf_token is not None}")

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("tokenizer loaded")

    # 2) Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )

    print(f"model_class={type(model)}")
    print(f"has_to={hasattr(model, 'to')}")

    model = model.to(device)
    model.eval()

    print("model loaded and moved to device")

    # 3) Tiny forward pass
    prompt = "The capital of France is"
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )

    print(f"logits_shape={tuple(outputs.logits.shape)}")
    print("GEMMA LOAD TEST PASSED")


if __name__ == "__main__":
    main()