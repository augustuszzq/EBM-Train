#!/usr/bin/env python3
"""DRL-backbone wrapper for the synchronized DDP/strict-pipeline trainer.

The canonical trainer is imported unchanged. This entrypoint replaces only its
model factory so the existing objective, replay, sampler, and pipeline semantics
remain the same.
"""

from __future__ import annotations

try:
    import ebm_train_sync_mode_a as _trainer
    from drl_resnet_energy import build_drl_resnet_energy_from_config
except Exception:  # pragma: no cover - package import path used by tests
    from polaris_ebm.scripts.current import ebm_train_sync_mode_a as _trainer
    from polaris_ebm.scripts.current.drl_resnet_energy import build_drl_resnet_energy_from_config


def _model_section(runtime) -> dict:
    cfg = getattr(runtime, "config", None)
    raw = getattr(cfg, "raw", {}) or {}
    model_raw = raw.get("model", {}) if isinstance(raw, dict) else {}
    return dict(model_raw or {})


def build_drl_energy_model(runtime, n_f: int, unconditional_cls=None):
    del n_f, unconditional_cls
    return build_drl_resnet_energy_from_config(_model_section(runtime), image_size=runtime.image_shape[-1])


def main() -> None:
    _trainer.build_energy_model = build_drl_energy_model
    _trainer.main()


if __name__ == "__main__":
    main()
