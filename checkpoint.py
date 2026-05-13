"""Checkpoint save/load for nets + optimizer + training step."""

import os
import torch


_LIVE_KEYS = ("representation", "dynamics", "policy", "value", "projector", "predictor")
_TARGET_KEYS = tuple(f"target_{k}" for k in _LIVE_KEYS)


def save_checkpoint(path: str, learner, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "step": step,
        "optimizer": learner.optimizer.state_dict(),
    }
    for k in _LIVE_KEYS:
        state[k] = getattr(learner.nets, k).state_dict()
    for k in _LIVE_KEYS:
        state[f"target_{k}"] = getattr(learner.target_nets, k).state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic; partial files never seen


def load_checkpoint(path: str, learner) -> int:
    state = torch.load(path, map_location=learner.device)
    for k in _LIVE_KEYS:
        getattr(learner.nets, k).load_state_dict(state[k])
    for k in _LIVE_KEYS:
        getattr(learner.target_nets, k).load_state_dict(state[f"target_{k}"])
    learner.optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [
        f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f[len("ckpt_") : -len(".pt")]))
    return os.path.join(ckpt_dir, candidates[-1])
