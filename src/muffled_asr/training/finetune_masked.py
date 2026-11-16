"""Fine-tune the Nemotron streaming checkpoint on masked speech.

This mirrors NeMo's own examples/asr/speech_to_text_finetune.py, with one
addition that matters on a consumer card: it turns on the joint network's fused
loss/WER path.

That step cannot be done from the config. NeMo's fine-tune script applies only
`model.train_ds`, `model.validation_ds`, `model.test_ds`, `model.tokenizer`,
`model.char_labels`, `model.optim` and `model.spec_augment` to the restored
model -- a `model.joint.*` block in the YAML is read by Hydra and then silently
ignored. Enabling it takes a method call on the restored joint, which is what
this script exists to do.

Why it matters: an RNN-T joint materialises a [B, T, U, V] tensor. At batch 8,
16 s of audio and a 1025-token vocabulary that is several GB on its own, and it
is the usual reason a transducer fine-tune OOMs on 24 GB. The fused path splits
that tensor into `fused_batch_size` sub-batches and frees each one after its
loss contribution is computed.

Everything about the encoder is left alone on purpose. The four runtime chunk
sizes (80/160/560/1120 ms) come from the encoder's att_context_size, and
overriding any of it collapses the model to a single latency operating point.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def disable_cuda_graph_decoder(asr_model: ASRModel) -> None:
    """Turn off CUDA graph capture in the RNN-T greedy decoder.

    NeMo defaults `use_cuda_graph_decoder` to True. The capture reserves a large
    static buffer the moment decoding is first used, which is at the end of the
    first training epoch — so the failure lands after a full epoch of apparently
    healthy training, which makes it look like anything but a memory problem.
    """
    decoding_cfg = getattr(asr_model.cfg, "decoding", None)
    if decoding_cfg is None or "greedy" not in decoding_cfg:
        logging.info("No greedy decoding config found; leaving CUDA graphs alone.")
        return

    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = False
    asr_model.change_decoding_strategy(decoding_cfg, verbose=False)
    logging.info("CUDA graph decoder disabled.")


def enable_fused_loss(asr_model: ASRModel, fused_batch_size: int) -> None:
    """Split the joint's [B, T, U, V] tensor so it fits in consumer VRAM."""
    if not hasattr(asr_model, "joint"):
        logging.warning("Model has no joint network; skipping fused loss (is this a CTC model?)")
        return

    asr_model.joint.set_fuse_loss_wer(True, asr_model.loss, asr_model.wer)
    asr_model.joint.set_fused_batch_size(fused_batch_size)
    logging.info(f"Joint fused loss/WER enabled, fused_batch_size={fused_batch_size}")


@hydra_runner(config_path="conf", config_name="finetune_masked")
def main(cfg: DictConfig) -> None:
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    fraction = cfg.get("gpu_mem_fraction")
    if fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(
            f"VRAM capped at {fraction:.0%} of {total_gib:.1f} GiB "
            f"({fraction * total_gib:.1f} GiB); the rest stays free for the display."
        )

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.get("init_from_nemo_model") is None:
        raise SystemExit("init_from_nemo_model must point at the .nemo checkpoint to fine-tune")

    asr_model = ASRModel.restore_from(restore_path=cfg.init_from_nemo_model)
    asr_model.set_trainer(trainer)
    logging.info("Reusing the tokenizer from the pre-trained model.")

    dataset_cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(dataset_cfg.model.train_ds)

    if dataset_cfg.model.validation_ds.get("manifest_filepath"):
        asr_model.setup_multiple_validation_data(dataset_cfg.model.validation_ds)
    else:
        # Training runs fine without it, but nothing can then rank the epochs:
        # no val_wer means no early stopping, and the exported .nemo is the last
        # epoch rather than the best one.
        logging.warning("No validation manifest — every epoch is kept, none is selected.")

    asr_model.setup_optimization(cfg.model.optim)

    if cfg.model.get("spec_augment") is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    if not cfg.get("cuda_graph_decoder", False):
        disable_cuda_graph_decoder(asr_model)

    if cfg.model.get("fuse_loss_wer", True):
        enable_fused_loss(asr_model, cfg.model.get("fused_batch_size", 4))

    if cfg.model.get("freeze_encoder", False):
        # Cheapest option when the dataset is small: adapt the decoder and joint
        # only, leaving the 600M-parameter encoder untouched.
        asr_model.encoder.freeze()
        logging.info("Encoder frozen — training decoder and joint only.")

    trainable = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in asr_model.parameters())
    logging.info(f"Trainable parameters: {trainable / 1e6:.1f}M of {total / 1e6:.1f}M")

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
