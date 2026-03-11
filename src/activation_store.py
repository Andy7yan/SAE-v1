import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HOOK_LAYER_INDEX, MAX_SEQ_LEN, MODEL_NAME, get_hf_token


class FrozenActivationModel:
    """
    Loads the frozen base model and captures one hidden activation tensor
    from a chosen layer using a forward hook.
    """

    def __init__(self, device: torch.device, model_dtype: torch.dtype):
        self.device = device
        self.model_dtype = model_dtype
        self.hf_token = get_hf_token()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=self.hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=self.hf_token,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.activation_store: dict[str, torch.Tensor] = {}

        def hook_fn(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Hook output is not a tensor: {type(output)}")

            self.activation_store["act"] = output.detach()

        target_module = self.model.model.layers[HOOK_LAYER_INDEX]
        self.handle = target_module.register_forward_hook(hook_fn)

    def capture_text_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.activation_store.clear()

        with torch.no_grad():
            _ = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )

        if "act" not in self.activation_store:
            raise RuntimeError("Hook fired zero times; no activation was captured.")

        act = self.activation_store["act"]
        attention_mask = batch["attention_mask"]
        return act, attention_mask

    def close(self) -> None:
        self.handle.remove()