#!/usr/bin/env python3
"""
Model/tokenizer loading for Gemma ONNX (CPU) runtimes.

Uses pdf2chemicals' runtime approach (GemmaChatTokenizer + manual ONNX
decode loop, no AutoTokenizer/AutoProcessor) to avoid schema mismatches in
tokenizer_config.json across transformers versions.

This module only knows about loading and raw text generation -- no
task-specific logic. `get_runtime()` caches a single `GemmaOnnxRuntime`
instance per process, so the model/tokenizer/ONNX sessions are loaded once
and reused everywhere.
"""

from __future__ import annotations

import json
import os
import sys

import jinja2
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer
from transformers import GenerationConfig

DEFAULT_MODEL_ID = "onnx-community/gemma-4-E4B-it-ONNX"
DEFAULT_QUANTIZATION = "q4"
DEFAULT_MAX_NEW_TOKENS = 4

onnxruntime.preload_dlls()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class GemmaChatTokenizer:
    """Thin wrapper around `tokenizers.Tokenizer` + Jinja2 chat template.

    Avoids AutoTokenizer/AutoProcessor, which can fail on these repos due to
    a schema mismatch in tokenizer_config.json (extra_special_tokens as a
    list instead of a dict) across some transformers versions.
    """

    def __init__(self, model_id: str):
        tokenizer_path = hf_hub_download(model_id, "tokenizer.json")
        config_path = hf_hub_download(model_id, "tokenizer_config.json")

        self._tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(config_path, "r", encoding="utf-8") as handle:
            self._config = json.load(handle)

        self.bos_token = self._config.get("bos_token", "")
        self.eos_token = self._config.get("eos_token", "")
        self.pad_token = self._config.get("pad_token", "")

        chat_template = self._config.get("chat_template")
        if not chat_template:
            raise ValueError("tokenizer_config.json has no chat_template.")

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        self._chat_template = env.from_string(chat_template)

        eos_id = self._tokenizer.token_to_id(self.eos_token)
        self.eos_token_id = [eos_id] if eos_id is not None else []

    def apply_chat_template(self, messages):
        return self._chat_template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
        )

    def encode_ids(self, text: str):
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode_ids(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class GemmaOnnxRuntime:
    """Loads a Gemma ONNX model/tokenizer once and generates raw text.

    Independently instantiable (e.g. for tests or a non-default instance),
    but the process-wide singleton should go through `get_instance()`
    (or the module-level `get_runtime()` convenience wrapper).
    """

    _instance: "GemmaOnnxRuntime | None" = None

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = DEFAULT_QUANTIZATION,
    ):
        self.model_id = model_id
        self.quantization = quantization
        self.tokenizer = GemmaChatTokenizer(model_id)
        self.eos_token_id = self._resolve_eos_token_id()
        self.embed_session, self.decoder_session = self._load_onnx_sessions()

    def _resolve_eos_token_id(self):
        try:
            generation_config = GenerationConfig.from_pretrained(self.model_id)
            return generation_config.eos_token_id
        except Exception:
            return self.tokenizer.eos_token_id

    def _load_onnx_sessions(self):
        embed_model = f"onnx/embed_tokens_{self.quantization}.onnx"
        decoder_model = f"onnx/decoder_model_merged_{self.quantization}.onnx"

        eprint(f"[{self.model_id}] Downloading/checking ONNX files ({self.quantization}) ...")
        model_dir = snapshot_download(
            self.model_id,
            allow_patterns=[f"{embed_model}*", f"{decoder_model}*"],
        )
        available_providers = onnxruntime.get_available_providers()
        eprint(available_providers)
        
        providers = ["CUDAExecutionProvider"]
        embed_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, embed_model), providers=providers
        )
        decoder_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, decoder_model), providers=providers
        )
        return embed_session, decoder_session

    def generate(self, messages, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        text = self.tokenizer.apply_chat_template(messages)
        ids = self.tokenizer.encode_ids(text)

        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        position_ids = (np.cumsum(attention_mask, axis=-1) - 1).astype(np.int64)
        batch_size = input_ids.shape[0]
        num_logits_to_keep = np.array(1, dtype=np.int64)
        past_key_values = {
            inp.name: np.zeros(
                [batch_size, inp.shape[1], 0, inp.shape[3]],
                dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
            )
            for inp in self.decoder_session.get_inputs()
            if inp.name.startswith("past_key_values")
        }

        eos_ids = (
            self.eos_token_id
            if isinstance(self.eos_token_id, (list, tuple))
            else [self.eos_token_id]
        )

        generated_tokens = np.empty((batch_size, 0), dtype=np.int64)
        for _ in range(max_new_tokens):
            inputs_embeds, per_layer_inputs = self.embed_session.run(
                None, {"input_ids": input_ids}
            )
            logits, *present_key_values = self.decoder_session.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "per_layer_inputs": per_layer_inputs,
                    "position_ids": position_ids,
                    "num_logits_to_keep": num_logits_to_keep,
                    **past_key_values,
                },
            )
            next_token = logits[:, -1].argmax(-1, keepdims=True).astype(np.int64)
            generated_tokens = np.concatenate([generated_tokens, next_token], axis=-1)
            if np.isin(next_token, eos_ids).any():
                break
            input_ids = next_token
            attention_mask = np.concatenate(
                [attention_mask, np.ones_like(next_token, dtype=np.int64)], axis=-1
            )
            position_ids = (position_ids[:, -1:] + 1).astype(np.int64)
            for index, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[index]

        return self.tokenizer.decode_ids(generated_tokens[0].tolist()).strip()

    @classmethod
    def get_instance(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = DEFAULT_QUANTIZATION,
    ) -> "GemmaOnnxRuntime":
        """Return the process-wide instance, creating it on first call.

        Subsequent calls (even with different arguments) return the
        already-loaded instance -- model/tokenizer/ONNX sessions load once.
        """
        if cls._instance is None:
            eprint(f"[{model_id}] Loading tokenizer, config and ONNX sessions ...")
            cls._instance = cls(model_id, quantization)
        return cls._instance


def get_runtime(
    model_id: str = DEFAULT_MODEL_ID,
    quantization: str = DEFAULT_QUANTIZATION,
) -> GemmaOnnxRuntime:
    """Convenience wrapper around `GemmaOnnxRuntime.get_instance()`."""
    return GemmaOnnxRuntime.get_instance(model_id, quantization)
