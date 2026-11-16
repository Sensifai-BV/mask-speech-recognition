"""Train encoder adapters on masked + noisy speech.

This replaces NeMo's examples/asr/asr_adapters/train_asr_adapter.py, which does
not work against this checkpoint. Its update_model_cfg() reconciles the user's
data config with the model's stored one by dropping every key the stored config
lacks, whitelisting only num_workers and pin_memory. This checkpoint stores a
Lhotse-style data config with no `batch_size`, so batch_size is dropped and the
loader then fails with "Missing key batch_size". Passing the stored config
through instead hits a second problem: it carries `num_buckets: null` and
`bucket_buffer_size: null`, while LhotseDataLoadingConfig declares both as plain
`int`.

The adapter mechanics below are the same as NeMo's -- rewrite the encoder target
to its adapter-capable subclass, add the adapter, freeze everything, then
unfreeze only the adapter. The data config is simply ours, unpruned.

Why the encoder: channel and noise robustness live in the encoder's early
layers. The decoder and joint model language, which is why an earlier run that
trained only those learned nothing useful from masked audio.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

LINEAR_ADAPTER = "nemo.collections.common.parts.adapter_modules.LinearAdapter"
RESIDUAL_STRATEGY = "nemo.core.classes.mixins.adapter_mixin_strategies.ResidualAddAdapterStrategy"


def make_encoder_adapter_capable(model_cfg: DictConfig) -> None:
    """Point the encoder at its adapter-supporting subclass before restore.

    Adapters can only be added to a module whose class implements the adapter
    mixin, and the base checkpoint names the plain class. This has to happen
    before the weights load, which is why the model is restored twice.
    """
    with open_dict(model_cfg):
        metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if metadata is None:
            raise SystemExit(
                f"No adapter-capable class registered for {model_cfg.encoder._target_}. "
                "This encoder does not support adapters in this NeMo version."
            )
        model_cfg.encoder._target_ = metadata.adapter_class_path


def build_adapter_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": LINEAR_ADAPTER,
            "in_features": cfg.model.adapter.in_features,
            "dim": cfg.model.adapter.dim,
            "activation": "swish",
            "norm_position": "pre",
            "dropout": cfg.model.adapter.dropout,
            "adapter_strategy": {
                "_target_": RESIDUAL_STRATEGY,
                "stochastic_depth": 0.0,
                "l2_lambda": 0.0,
            },
        }
    )


def slim_checkpoints(model) -> None:
    """Keep frozen base weights out of every Lightning checkpoint.

    A full checkpoint of this model is ~2.4 GB of frozen weights plus ~7 MB that
    actually changes (adapters + their optimizer state). Lightning serialises a
    checkpoint into an in-memory buffer before writing it, so each save briefly
    needs about twice that in host RAM, and that spike is what exhausted memory
    on the laptop. Frozen *parameters* are dropped; buffers (e.g. BatchNorm
    running stats, which still update in train mode) are kept, so resume is exact.

    Loading: the frozen weights already come from init_from_nemo_model, so
    restore (resume, and save_best_model at train end) runs with
    strict_loading=False; old full checkpoints still load. A checkpoint without
    the trainable tensors is refused instead of silently training from scratch.
    """
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    orig_save, orig_load = model.on_save_checkpoint, model.on_load_checkpoint

    def on_save_checkpoint(checkpoint):
        out = orig_save(checkpoint)
        sd = checkpoint["state_dict"]
        for k in [k for k in sd if k in frozen]:
            del sd[k]
        checkpoint["slim_checkpoint"] = True
        return out

    def on_load_checkpoint(checkpoint):
        missing = trainable - set(checkpoint.get("state_dict", {}))
        if missing:
            raise SystemExit(f"checkpoint lacks {len(missing)} trainable tensors, e.g. {sorted(missing)[:2]}")
        return orig_load(checkpoint)

    model.on_save_checkpoint = on_save_checkpoint
    model.on_load_checkpoint = on_load_checkpoint
    model.strict_loading = False
    logging.info(f"Slim checkpoints: {len(frozen)} frozen parameter tensors excluded, {len(trainable)} trainable kept")


@hydra_runner(config_path="conf", config_name="adapter_masked")
def main(cfg: DictConfig) -> None:
    # Sub-trees of this config get copied into the model's own config, where
    # the `model.` root no longer exists. Resolve now so no interpolation has
    # to survive that move.
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    fraction = cfg.get("gpu_mem_fraction")
    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"VRAM capped at {fraction:.0%} of {total:.1f} GiB")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.get("init_from_nemo_model") is None:
        raise SystemExit("init_from_nemo_model must point at the .nemo checkpoint")

    model_cfg = ASRModel.restore_from(cfg.init_from_nemo_model, return_config=True)
    make_encoder_adapter_capable(model_cfg)
    with open_dict(model_cfg):
        model_cfg.log_prediction = cfg.model.get("log_prediction", False)
    model = ASRModel.restore_from(
        cfg.init_from_nemo_model, override_config_path=model_cfg, trainer=trainer
    )

    encoder_dim = model.cfg.encoder.get("d_model")
    if encoder_dim and encoder_dim != cfg.model.adapter.in_features:
        raise SystemExit(
            f"adapter.in_features is {cfg.model.adapter.in_features} but the encoder's "
            f"d_model is {encoder_dim}. A mismatch fails later with an opaque shape error."
        )

    # Our data config, passed through unmodified. This is the whole reason this
    # script exists rather than NeMo's.
    model.setup_training_data(cfg.model.train_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    model.setup_optimization(cfg.model.optim)

    if cfg.model.get("spec_augment") is not None:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)

    adapter_name = f"{cfg.model.adapter.module}:{cfg.model.adapter.name}"
    model.add_adapter(adapter_name, cfg=build_adapter_cfg(cfg))
    if not model.is_adapter_available():
        raise SystemExit("adapter was not registered on the model")

    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)

    # Freeze the whole 600M model, then unfreeze only the adapters. This is what
    # keeps optimizer state small enough to train the encoder on 8 GB.
    model.freeze()
    model = model.train()
    model.unfreeze_enabled_adapters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Adapter '{adapter_name}' — trainable {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
        f"({100 * trainable / total:.2f}%)"
    )
    if trainable == 0:
        raise SystemExit("nothing is trainable — the adapter did not attach")

    if cfg.model.get("slim_checkpoints", True):
        slim_checkpoints(model)

    if cfg.model.get("fuse_loss_wer", True):
        fused = cfg.model.get("fused_batch_size", 2)
        model.joint.set_fuse_loss_wer(True, model.loss, model.wer)
        model.joint.set_fused_batch_size(fused)
        logging.info(f"Joint fused loss/WER enabled, fused_batch_size={fused}")

    # The WER metric reads this at call time, so set it on the live object too.
    model.wer.log_prediction = cfg.model.get("log_prediction", False)

    model.cfg = model.cfg
    trainer.fit(model)

    # With save_best_model, NeMo's on_train_end has already restored the best
    # checkpoint into `model` at this point, so these are the best weights.
    if cfg.model.adapter.get("state_dict_name"):
        path = f"{trainer.log_dir}/{cfg.model.adapter.state_dict_name}"
        model.save_adapters(path)
        logging.info(f"Adapter weights saved to {path}")


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
