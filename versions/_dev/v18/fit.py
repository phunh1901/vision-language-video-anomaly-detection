# ruff: noqa: F821
"""Injected into the frozen-checkpoint trainer; consumes only train/development."""

from head import TEMPORAL_LAGS, TemporalHead, anomaly_logits
from v7_windows import overlap_forward


def cache_v7(model, context, indices, make_mask):
    pack, ids, labels, _, args, device, prompts, _ = context
    cache = {}
    width = args.visual_length
    checked = 0
    with torch.inference_mode():
        for i in indices:
            key = ids[i]
            item = pack.features[key]
            array = pack.array(item)
            steps = len(array)
            blocks = (steps + width - 1) // width
            x = torch.zeros((blocks * width, 512), device=device)
            x[:steps] = torch.as_tensor(array, dtype=torch.float32, device=device)
            lengths = (steps - torch.arange(blocks) * width).clamp(max=width)
            _, c, a = overlap_forward(model, x.reshape(blocks, width, 512),
                                      make_mask(lengths, width).to(device), prompts, lengths,
                                      make_mask)
            c, binary_a = anomaly_logits(c, a)
            allowed = labels[i, 1:].bool().to(device)
            logits = a.float().reshape(-1, a.shape[-1])[:steps]
            semantic = (logits[:, 1:][:, allowed].logsumexp(-1) - logits[:, 0]
                        if allowed.any() else binary_a.reshape(-1)[:steps])
            teacher = (c.reshape(-1)[:steps].sigmoid() + semantic.sigmoid()) / 2
            neighbor = teacher if steps == 1 else torch.maximum(
                F.pad(teacher[:-1], (1, 0)), F.pad(teacher[1:], (0, 1)))
            base = (c if PREFIX == "ucf" else binary_a).reshape(-1)[:steps]
            if not torch.isfinite(base).all() or not torch.isfinite(teacher).all():
                raise ValueError(f"Invalid VadCLIP logits: {key}")
            cache[int(i)] = (array.astype(np.float16), base.cpu().numpy(),
                             torch.minimum(teacher, neighbor).cpu().numpy())
            checked += 1
            if checked % 100 == 0 or checked == len(indices):
                print(f"{PREFIX}: cached V7 + CLIP {checked}/{len(indices)}", flush=True)
    return cache, {"videos": checked, "temporal_lags": TEMPORAL_LAGS}


def candidates(teacher, positive, rng):
    # Explicitly skip ambiguous positives and distinct isolated high peaks.
    if positive:
        valid = np.flatnonzero(teacher >= 0.50)
        rank = valid[np.argsort(-teacher[valid], kind="stable")]
        return rank[:min(16, max(1, len(teacher) // 16))] if len(teacher) > 1 else rank[:0]
    return rng.choice(len(teacher), min(16, len(teacher)), replace=False)


def fit_one(context, schedule, dev, cache, variant):
    _, _, _, abnormal, _, device, _, _ = context
    seed_all(SEED)
    head = TemporalHead().to(device)
    use_temporal = variant != "clip_only"
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    total_pairs = 0
    history = []
    for epoch, batches in enumerate(schedule):
        head.train()
        rng = np.random.default_rng(SEED + epoch)
        pair_rng = np.random.default_rng(SEED + len(schedule) + epoch)
        losses = []
        for batch in batches:
            scores, pool = [], {False: [], True: []}
            for i in batch:
                clip, base, teacher = cache[int(i)]
                x = torch.as_tensor(clip, dtype=torch.float32, device=device)
                b = torch.as_tensor(base, device=device)
                signal = head(x, use_temporal)
                z = b + signal
                scores.append(z.topk(1 + len(z) // 16).values.mean())
                pos = bool(abnormal[i])
                ix = candidates(teacher, pos, rng)
                if len(ix):
                    ix = torch.as_tensor(ix, device=device)
                    pool[pos].append((x[ix].detach(), z[ix]))
            labels = torch.as_tensor(abnormal[batch], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(torch.stack(scores), labels)
            if pool[True] and pool[False]:
                px, pz = (torch.cat(items) for items in zip(*pool[True], strict=True))
                nx, nz = (torch.cat(items) for items in zip(*pool[False], strict=True))
                if variant == "v18":
                    match = (F.normalize(px, dim=-1) @ F.normalize(nx, dim=-1).T).argmax(1)
                else:
                    match = torch.as_tensor(pair_rng.integers(len(nz), size=len(pz)), device=device)
                loss = loss + 0.5 * F.softplus(0.5 - pz + nz[match]).mean()
                total_pairs += len(pz)
            if not torch.isfinite(loss):
                raise ValueError(f"Invalid V18 training loss: {variant}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f"{PREFIX}/{variant}: epoch {epoch+1}, loss={history[-1]:.5f}", flush=True)
    head.eval()
    strengths = (0., 0.25, 0.5, 1.)
    dev_scores = {strength: [] for strength in strengths}
    with torch.no_grad():
        for i in dev:
            clip, base, _ = cache[int(i)]
            x = torch.as_tensor(clip, dtype=torch.float32, device=device)
            delta = head(x, use_temporal)
            b = torch.as_tensor(base, device=device)
            for strength in strengths:
                dev_scores[strength].append(float((b + strength * delta).topk(1 + len(b)//16).values.mean()))
    curve = {s: float(average_precision_score(abnormal[dev], scores))
             for s, scores in dev_scores.items()}
    strength = max(strengths, key=curve.get)
    path = OUTPUT / f"{variant}.pth"
    save_checkpoint(head, path)
    return {"strength": strength, "dev_video_ap": curve, "paired_examples": total_pairs,
            "epoch_loss": history, "checkpoint_file": path.name, "checkpoint_sha256": file_sha(path)}


def fit_variants(model, context, train, dev, make_mask):
    _, _, _, abnormal, _, _, _, _ = context
    indices = np.unique(np.r_[train, dev])
    schedule = [balanced_batches(train, abnormal, 2 if SMOKE_ONLY else HALF_BATCH, SEED + epoch)
                for epoch in range(1 if SMOKE_ONLY else TEMPORAL_CONFIG["epochs"])]
    if SMOKE_ONLY:
        schedule = [schedule[0][:2]]
        indices = np.unique(np.r_[schedule[0].reshape(-1), dev])
    cache, receipt = cache_v7(model, context, indices, make_mask)
    result = {variant: fit_one(context, schedule, dev, cache, variant) for variant in VARIANTS}
    if not SMOKE_ONLY and any(r["paired_examples"] == 0 for r in result.values()):
        raise ValueError("No valid positive-negative pairs. Inspect training candidate coverage.")
    return result, receipt
