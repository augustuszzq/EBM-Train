#!/usr/bin/env python3
"""Generate samples from a DRL-style ImageNet-32 EBM checkpoint.

This is a thin wrapper around `eval_generate.py`. It replaces only the model
factory so ImageNet-32 DRL checkpoints are reconstructed with
`DRLResNetEnergy`, not the default conditional model.
"""

from __future__ import annotations

try:
    import eval_generate as _eval_generate
    from drl_resnet_energy import build_drl_resnet_energy_from_config
except Exception:  # pragma: no cover - package import path used by tests
    from polaris_ebm.scripts.current import eval_generate as _eval_generate
    from polaris_ebm.scripts.current.drl_resnet_energy import build_drl_resnet_energy_from_config


def _model_section(runtime) -> dict:
    cfg = getattr(runtime, "config", None)
    raw = getattr(cfg, "raw", {}) or {}
    model_raw = raw.get("model", {}) if isinstance(raw, dict) else {}
    return dict(model_raw or {})


def build_drl_eval_model(runtime, n_f: int, unconditional_cls=None):
    del n_f, unconditional_cls
    return build_drl_resnet_energy_from_config(_model_section(runtime), image_size=runtime.image_shape[-1])


def main() -> None:
    _eval_generate.build_energy_model = build_drl_eval_model
    _eval_generate.main()


if __name__ == "__main__":
    main()
