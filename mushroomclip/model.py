from transformers import Trainer, TrainingArguments, CLIPModel, CLIPProcessor
import torch.nn as nn
import torch.nn.functional as F


from os import cpu_count
from pathlib import Path


def load_model(name: str = "") -> tuple[CLIPModel, CLIPProcessor]:
    default_name = "openai/clip-vit-base-patch32"
    if name is None or len(name) == 0:
        name = default_name
        
    model = CLIPModel.from_pretrained(name)
    processor = CLIPProcessor.from_pretrained(default_name)

    return model, processor


class CLIPMushroomTrainer(Trainer):
    def compute_loss(self, model, batch, return_outputs=False):
        labels = batch.pop("labels")
        outputs = model(**batch)
        logits_im = outputs.get("logits_per_image")
        logits_te = outputs.get("logits_per_text")

        def loss_fn(logits, target):
            logprobs = F.log_softmax(logits, dim=1)
            loss = - torch.sum(target * logprobs, dim=1)
            return torch.mean(loss)

        loss = (loss_fn(logits_im, labels) + loss_fn(logits_te, labels)) / 2
        if return_outputs:
            return (loss, outputs)
        return loss


def get_train_args(
    checkpoint_dir: Path,
    batch_size: int,
    accum_steps: int,
    learning_rate: float,
    epochs: int,
    log_every: int,
    save_every: int,
    warmup_ratio: float = 0.05,
    fp16: bool = False
    ):
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        evaluation_strategy="no",
        prediction_loss_only=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        dataloader_num_workers=cpu_count(),
        dataloader_drop_last=True,
        load_best_model_at_end=False,
        ignore_data_skip=True,
        resume_from_checkpoint=str(checkpoint_dir),
        logging_steps=log_every,
        save_steps=save_every,
    )

    
