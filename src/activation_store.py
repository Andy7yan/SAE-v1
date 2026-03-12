import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_SEQ_LEN, MODEL_NAME, TOKEN_LENGTH_PROBE_MAX_LEN
from dist_utils import log0


class StopForward(Exception):
    pass


class FrozenActivationModel:
    def __init__(self, device: torch.device, model_dtype: torch.dtype, hook_layer_index: int):
        self.device = device
        self.hook_layer_index = hook_layer_index
        self.token = os.environ.get("HF_TOKEN")

        log0("Loading tokeniser ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=self.token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = {
            token_id for token_id in self.tokenizer.all_special_ids if token_id is not None and token_id >= 0
        }

        log0(f"Loading {MODEL_NAME} (frozen) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=self.token,
            torch_dtype=model_dtype,
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

        self.handle = self.model.model.layers[hook_layer_index].register_forward_hook(hook_fn)
        log0(f"Residual-stream hook registered at layer {hook_layer_index}")

    def tokenize_text_batch(self, texts: list[str]):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        return batch

    def build_token_mask(self, batch) -> torch.Tensor:
        mask = batch["attention_mask"].bool()
        input_ids = batch["input_ids"]

        for token_id in self.special_token_ids:
            mask &= input_ids != token_id

        return mask

    def length_stats(self, texts: list[str]):
        raw_batch = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max(MAX_SEQ_LEN, TOKEN_LENGTH_PROBE_MAX_LEN),
            return_length=True,
        )
        raw_lengths = raw_batch["length"]

        truncated_batch = self.tokenize_text_batch(texts)
        valid_mask = self.build_token_mask(truncated_batch)
        valid_lengths = valid_mask.sum(dim=-1).tolist()
        truncation_count = sum(1 for length in raw_lengths if length > MAX_SEQ_LEN)

        return {
            "raw_lengths": raw_lengths,
            "valid_lengths": valid_lengths,
            "truncation_count": truncation_count,
        }

    @torch.no_grad()
    def capture_text_batch(self, texts: list[str]):
        batch = self.tokenize_text_batch(texts)
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
        mask = self.build_token_mask(batch)
        return act, mask

    def close(self):
        self.handle.remove()