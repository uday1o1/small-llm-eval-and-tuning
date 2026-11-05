"""Parameters per component and inference-state memory of a hybrid-attention model and a
full-attention model of similar size, from their config.json files (GPU for the measured part).

    python scripts/architecture.py results/architecture

1. Each model's config.json is fetched at a pinned Hub commit (MODELS) and the language model is built on
   PyTorch's `meta` device (no weights), and its parameters are counted by component.
2. The same counts are taken from the tensor shapes in the published safetensors headers, as a
   check that the meta-device model matches the released weights.
3. The memory the model keeps per sequence (KV cache of the full-attention layers, plus the fixed
   recurrent and convolution states of the linear-attention layers) is computed from the config
   and measured: the cache object returned by a bf16 forward pass of T tokens is summed tensor by
   tensor, for several T.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

MODELS = {"Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",     # Hub commits
          "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}
MEASURE_T = [1024, 4096, 16384, 32768]
COMPONENTS = ["embedding", "lm_head", "full_attention", "linear_attention", "mlp", "norm"]


def component(name: str) -> str | None:
    if "visual" in name or name.startswith("mtp"):
        return None                                   # vision encoder and MTP head: not the language model
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if ".self_attn." in name:
        return "full_attention"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    raise ValueError(name)


def count(named_shapes) -> dict:
    out = dict.fromkeys(COMPONENTS, 0)
    for name, shape in named_shapes:
        c = component(name)
        if c:
            n = 1
            for s in shape:
                n *= s
            out[c] += n
    return out


def state_bytes(obj, seen=None, depth=0) -> int:
    """Total bytes of the distinct tensors reachable from a cache object."""
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        if obj.data_ptr() in seen or obj.numel() == 0:
            return 0
        seen.add(obj.data_ptr())
        return obj.numel() * obj.element_size()
    if depth > 6:
        return 0
    if isinstance(obj, (list, tuple)):
        return sum(state_bytes(o, seen, depth + 1) for o in obj)
    if isinstance(obj, dict):
        return sum(state_bytes(o, seen, depth + 1) for o in obj.values())
    if hasattr(obj, "__dict__"):
        return sum(state_bytes(o, seen, depth + 1) for o in vars(obj).values())
    return 0


def analytic(tc) -> dict:
    types = list(getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers)
    n_full = types.count("full_attention")
    n_lin = types.count("linear_attention")
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    kv_per_token = 2 * n_full * tc.num_key_value_heads * hd * 2              # K and V, bf16
    lin = {}
    if n_lin:
        dk, dv = tc.linear_key_head_dim, tc.linear_value_head_dim
        hk, hv = tc.linear_num_key_heads, tc.linear_num_value_heads
        conv_dim = 2 * hk * dk + hv * dv
        lin = {"recurrent_state_elems_per_layer": hv * dk * dv,                  # kept in fp32 by transformers
               "conv_state_elems_per_layer": conv_dim * tc.linear_conv_kernel_dim,  # transformers keeps the full kernel width
               "n_linear_layers": n_lin}
    return {"n_layers": len(types), "n_full_attention": n_full, "n_linear_attention": n_lin,
            "hidden": tc.hidden_size, "q_heads": tc.num_attention_heads, "kv_heads": tc.num_key_value_heads,
            "head_dim": hd, "vocab": tc.vocab_size, "tied_embeddings": bool(getattr(tc, "tie_word_embeddings", False)),
            "max_position_embeddings": tc.max_position_embeddings,
            "kv_bytes_per_token_bf16": kv_per_token,
            "attention_flops_per_token_per_context_token": 4 * n_full * tc.num_attention_heads * hd, **lin}


@torch.no_grad()
def measure(name, sha) -> list[dict]:
    model = AutoModelForCausalLM.from_pretrained(name, revision=sha, dtype=torch.bfloat16).cuda().eval()
    out = []
    for T in MEASURE_T:
        ids = torch.randint(1000, 20000, (1, T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
        o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        out.append({"T": T, "state_bytes": state_bytes(o.past_key_values)})
        del o
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, sha in MODELS.items():
        shutil.copy(hf_hub_download(name, "config.json", revision=sha), out / f"{name.replace('/', '__')}@{sha}.config.json")
        cfg = AutoConfig.from_pretrained(name, revision=sha)
        tc = getattr(cfg, "text_config", cfg)
        with init_empty_weights():                    # parameters on the meta device, no memory
            model = AutoModelForCausalLM.from_config(tc)
        meta = count((n, p.shape) for n, p in model.named_parameters())
        st = get_safetensors_metadata(name, revision=sha)
        shapes = [(n.replace("model.language_model.", "model."), t.shape)
                  for f in st.files_metadata.values() for n, t in f.tensors.items()]
        released = count(shapes)
        if analytic(tc)["tied_embeddings"]:
            meta["lm_head"] = released["lm_head"] = 0  # a tied head shares the embedding matrix
        res[name] = {"sha": sha, "model_class": type(model).__name__, "params_meta": meta,
                     "params_safetensors": released, "params_total_meta": sum(meta.values()),
                     "config": analytic(tc), "measured": measure(name, sha) if torch.cuda.is_available() else []}
        print(name, json.dumps(res[name]))
    (out / "architecture.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
