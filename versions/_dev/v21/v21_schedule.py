"""Two-stage warm-start policy; frozen CLIP, persistent Adam state across phases."""

import math

import torch


class TwoStageTraining:
    """Epoch indices are zero-based. Smoke executes one epoch of each phase."""

    mode = "two_stage_A"

    def __init__(self, model, config, smoke=False):
        self.model = model
        self.settings = dict(config["two_stage"])
        cfg = self.settings
        for key in ("warmup_epochs", "finetune_epochs"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["epochs"] != cfg["warmup_epochs"] + cfg["finetune_epochs"]:
            raise ValueError("epochs must equal warmup_epochs + finetune_epochs")
        for key in ("head_lr", "finetune_head_lr", "adapter_lr", "prompt_lr",
                    "curvature_lr", "scale_lr", "finetune_curvature_lr", "finetune_scale_lr"):
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"Invalid positive learning rate: {key}")
        if not 0 < cfg["end_lr_ratio"] <= 1:
            raise ValueError("end_lr_ratio must be in (0,1]")
        if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.warmup_epochs = 1 if smoke else cfg["warmup_epochs"]
        self.finetune_epochs = 1 if smoke else cfg["finetune_epochs"]
        self.epochs = self.warmup_epochs + self.finetune_epochs
        self.phase, self.epoch = None, -1
        buckets = {}
        self.names = {}
        for name, parameter in model.named_parameters():
            if name.startswith("base.clipmodel."):
                parameter.requires_grad_(False)
                continue
            if name == "geometry.raw_c":
                kind = "curvature"
            elif name in {"geometry.raw_v", "geometry.raw_t"}:
                kind = "scale"
            elif name.startswith("geometry."):
                kind = "head"
            elif model.parameter_group(name) == "prompt_fusion":
                kind = "prompt"
            else:
                kind = "adapter"
            # Scalars and bias/gain vectors are not decayed.
            decay = config["weight_decay"] if parameter.ndim >= 2 else 0.
            buckets.setdefault((kind, decay), []).append(parameter)
            self.names[name] = kind
        if not {"head", "curvature", "scale", "adapter", "prompt"} <= {
                kind for kind, _ in buckets}:
            raise ValueError("Missing required parameter groups for two-stage training")
        self.optimizer = torch.optim.Adam([
            {"params": parameters, "kind": kind, "lr": 0., "weight_decay": decay}
            for (kind, decay), parameters in buckets.items()
        ], lr=cfg["head_lr"])

    def begin_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < self.epochs:
            raise ValueError("Epoch outside two-stage schedule")
        self.epoch = epoch
        self.phase = "head_warmup" if epoch < self.warmup_epochs else "joint_finetune"
        cfg = self.settings
        if self.phase == "head_warmup":
            factor = (epoch + 1) / self.warmup_epochs
            rates = {"head": cfg["head_lr"], "curvature": cfg["curvature_lr"],
                     "scale": cfg["scale_lr"], "adapter": 0., "prompt": 0.}
        else:
            step = epoch - self.warmup_epochs
            progress = step / max(1, self.finetune_epochs - 1)
            factor = cfg["end_lr_ratio"] + (1 - cfg["end_lr_ratio"]) * (
                1 + math.cos(math.pi * progress)) / 2
            rates = {"head": cfg["finetune_head_lr"], "curvature": cfg["finetune_curvature_lr"],
                     "scale": cfg["finetune_scale_lr"], "adapter": cfg["adapter_lr"],
                     "prompt": cfg["prompt_lr"]}
        for name, parameter in self.model.named_parameters():
            kind = self.names.get(name)
            parameter.requires_grad_(kind is not None and rates[kind] > 0)
            parameter.grad = None
        for group in self.optimizer.param_groups:
            group["lr"] = rates[group["kind"]] * factor
        self.train_mode()
        return self.receipt()

    def train_mode(self):
        self.model.train()
        if self.phase == "head_warmup":
            # Freeze stochastic/stateful behavior too, not only weight gradients.
            self.model.base.eval()
        self.model.base.clipmodel.eval()

    def check_gradients(self, require_nonzero=True):
        active = set()
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if not parameter.requires_grad:
                raise RuntimeError(f"Frozen parameter received gradient: {name}")
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"Non-finite gradient: {name}")
            if bool(gradient.abs().sum() > 0):
                active.add(self.names[name])
        expected = {"head"} if self.phase == "head_warmup" else {"head", "adapter", "prompt"}
        if require_nonzero and not expected <= active:
            raise RuntimeError(f"Disconnected {self.phase} groups: {sorted(expected - active)}")
        # Curvature/scales may legitimately have zero gradients on individual batches.
        return {"phase": self.phase, "nonzero_groups": sorted(active)}

    def receipt(self):
        return {"phase": self.phase, "epoch": self.epoch + 1,
                "warmup_epochs": self.warmup_epochs, "finetune_epochs": self.finetune_epochs,
                "learning_rates": {g["kind"]: g["lr"] for g in self.optimizer.param_groups},
                "settings": self.settings}

    def state_dict(self):
        return self.receipt()


class FromScratchTraining(TwoStageTraining):
    """Train all new downstream weights; freeze c/scales during low-curvature warm-up."""

    mode = "from_scratch_B"

    def __init__(self, model, config, smoke=False):
        settings = dict(config["from_scratch"])
        if config.get("initialization") != "fresh_downstream":
            raise ValueError("B requires initialization=fresh_downstream")
        mapped = {**config, "two_stage": {
            "warmup_epochs": settings["warmup_epochs"],
            "finetune_epochs": config["epochs"] - settings["warmup_epochs"],
            "head_lr": config["learning_rate"],
            "finetune_head_lr": config["learning_rate"],
            "adapter_lr": config["learning_rate"], "prompt_lr": config["learning_rate"],
            "curvature_lr": settings["curvature_lr"], "scale_lr": settings["scale_lr"],
            "finetune_curvature_lr": settings["curvature_lr"],
            "finetune_scale_lr": settings["scale_lr"],
            "end_lr_ratio": settings["end_lr_ratio"],
        }}
        super().__init__(model, mapped, smoke)

    def begin_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < self.epochs:
            raise ValueError("Epoch outside B schedule")
        self.epoch = epoch
        warmup = epoch < self.warmup_epochs
        self.phase = "low_curvature_warmup" if warmup else "hyperbolic_joint"
        cfg = self.settings
        if warmup:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.finetune_epochs - 1)
            factor = cfg["end_lr_ratio"] + (1 - cfg["end_lr_ratio"]) * (
                1 + math.cos(math.pi * progress)) / 2
        for name, parameter in self.model.named_parameters():
            kind = self.names.get(name)
            parameter.requires_grad_(kind is not None and not (
                warmup and kind in {"curvature", "scale"}))
            parameter.grad = None
        for group in self.optimizer.param_groups:
            kind = group["kind"]
            rate = cfg["head_lr"] if kind in {"head", "adapter", "prompt"} else cfg[kind + "_lr"]
            group["lr"] = 0. if warmup and kind in {"curvature", "scale"} else rate * factor
        self.train_mode()
        return self.receipt()

    def train_mode(self):
        self.model.train()
        self.model.base.clipmodel.eval()
