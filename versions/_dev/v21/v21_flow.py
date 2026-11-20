"""Observe real encoder calls/gradients; never substitute a synthetic adapter at runtime."""

import hashlib
import inspect
import re

import torch


def _tensor_info(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [_tensor_info(item) for item in value]
    return type(value).__name__


class EncoderFlowAudit:
    """Single forward/backward trace on TRAIN data, removing all hooks on exit.

    Checks local Transformer -> all registered global GCN layers -> visual FC;
    learned text prompts -> frozen CLIP text Transformer -> text mapping.
    It also rejects accidental calls to CLIP's image encoder on cached features.
    The paper's exact window layout remains a separate, unverified source detail.
    """

    def __init__(self, model):
        self.model = model
        base = model.base
        names = [name for name, _ in base.named_children() if re.fullmatch(r"gc\d+", name)]
        if not names or not all(hasattr(base, name) for name in (
                "temporal", "linear", "text_prompt_embeddings")):
            raise ValueError("Flow audit requires the actual VadCLIP local/global/prompt modules")
        if not hasattr(base.clipmodel, "transformer"):
            raise ValueError("Flow audit requires the frozen CLIP text Transformer")
        self.modules = {"local_transformer": base.temporal,
                        **{f"global_{name}": getattr(base, name) for name in names},
                        "temporal_output_fc": base.linear,
                        "text_prompts": base.text_prompt_embeddings,
                        "clip_text_transformer": base.clipmodel.transformer}
        self.native = model.space == "native_cosine"
        if self.native:
            self.modules.update(native_classifier=base.classifier, native_classifier_ffn=base.mlp2,
                                native_prompt_fusion=base.mlp1)
        else:
            self.modules.update(visual_projection=model.geometry.visual,
                                text_mapping_input=model.geometry.text)
        if hasattr(base.clipmodel, "visual"):
            self.modules["clip_image_encoder"] = base.clipmodel.visual
        self.events, self.handles = [], []

    def __enter__(self):
        def hook(name):
            def record(module, args, output):
                self.events.append({"stage": name, "input": _tensor_info(args),
                                    "output": _tensor_info(output)})
            return record

        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(hook(name)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def receipt(self):
        stages = [event["stage"] for event in self.events]
        required = set(self.modules) - {"clip_image_encoder"}
        if required - set(stages):
            raise RuntimeError(f"Disconnected forward stages: {sorted(required - set(stages))}")
        if "clip_image_encoder" in stages:
            raise RuntimeError("Cached feature pipeline unexpectedly executed CLIP image encoder")
        global_stages = [name for name in required if name.startswith("global_")]
        before = [("local_transformer", name) for name in global_stages]
        before += [(name, "temporal_output_fc") for name in global_stages]
        before += [("text_prompts", "clip_text_transformer")]
        if self.native:
            before += [("temporal_output_fc", "native_classifier_ffn"),
                       ("native_classifier_ffn", "native_classifier"),
                       ("clip_text_transformer", "native_prompt_fusion")]
        else:
            before += [("temporal_output_fc", "visual_projection"),
                       ("clip_text_transformer", "text_mapping_input")]
        if any(stages.index(a) >= stages.index(b) for a, b in before):
            raise RuntimeError(f"Encoder order violates local/global/text mapping flow: {stages}")
        gradient_groups = {name: any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    and bool((p.grad != 0).any()) for p in module.parameters())
                           for name, module in self.modules.items()
                           if name not in {"clip_text_transformer", "clip_image_encoder",
                                           "text_mapping_input"}}
        nonfinite = [name for name, p in self.model.named_parameters()
                     if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        frozen_gradients = any(p.grad is not None for p in self.model.base.clipmodel.parameters())
        if nonfinite or frozen_gradients or not all(gradient_groups.values()):
            raise RuntimeError(f"Broken encoder gradient flow: {gradient_groups}; "
                               f"frozen_clip_gradients={frozen_gradients}; nonfinite={nonfinite}")
        source = {name: inspect.getsource(getattr(type(self.model.base), name))
                  for name in ("encode_video", "encode_textprompt")}
        if self.native:
            source["forward"] = inspect.getsource(self.model._native_forward)
        return {"verified_on": "one official training video; forward and backward",
                "test_data_used": False, "stages": self.events,
                "nonzero_finite_gradient_groups": gradient_groups,
                "frozen_clip_gradients": frozen_gradients,
                "image_encoder_called": False,
                "encoder_source": source,
                "encoder_source_sha256": {name: hashlib.sha256(value.encode()).hexdigest()
                                          for name, value in source.items()},
                "exact_poinclip_author_code_verified": False}
