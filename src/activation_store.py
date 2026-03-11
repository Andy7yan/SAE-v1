import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HOOK_LAYER_INDEX, MAX_SEQ_LEN, MODEL_NAME
from dist_utils import log0


class StopForward(Exception):
    pass


class FrozenActivationModel:
    def __init__(self, device: torch.device, model_dtype: torch.dtype):
        self.device = device
        self.token = os.environ.get("HF_TOKEN")

        log0("Loading tokeniser ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=self.token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log0(f"Loading {MODEL_NAME} (frozen) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=self.token,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.activation_store = {}

        def hook_fn(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Hook output is not a tensor: {type(output)}")
            self.activation_store["act"] = output.detach()
            raise StopForward()

        self.handle = self.model.model.layers[HOOK_LAYER_INDEX].register_forward_hook(hook_fn)
        log0(f"Residual-stream hook registered at layer {HOOK_LAYER_INDEX}")

    @torch.no_grad()
    def capture_text_batch(self, texts: list[str]):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.activation_store.clear()

        try:
            self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        except StopForward:
            pass

        if "act" not in self.activation_store:
            raise RuntimeError("Hook did not capture activation.")

        act = self.activation_store["act"].to(torch.float32)
        mask = batch["attention_mask"].bool()
        mask[:, 0] = False
        return act, mask

    def close(self):
        self.handle.remove()