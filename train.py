import mushroomclip as mclip
import typer

from datetime import datetime
from pathlib import Path


def train(
    images_dir: Path = "data/mushroom_images/",
    tsv_path: Path = "data/mushrooms.tsv.gz",
    checkpoint_dir: Path = "checkpoints/",
    test_split_date: datetime = "2020-01-01",
    batch_size: int = 128,
    accum_steps: int = 10,
    learning_rate: float = 5e-05,
    epochs: int = 10,
    resume: bool = False,
    save_every: int = 100,
    log_every: int = 10,
    fp16: bool = False
    ):
    data = mclip.load_data(tsv_path, test_split_date)

    trainset = mclip.MushroomDataset(data["train"], images_dir)
    testset = mclip.MushroomDataset(data["test"], images_dir, augment=False)
    model, processor = mclip.load_model()
    collator = mclip.CLIPMushroomCollate(processor)
    args = mclip.get_train_args(
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        accum_steps=accum_steps,
        learning_rate=learning_rate,
        epochs=epochs,
        log_every=log_every,
        save_every=save_every,
        fp16=fp16
    )

    trainer = mclip.CLIPMushroomTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=trainset,
        eval_dataset=testset
    )
    if resume:
        trainer.train(str(checkpoint_dir))
    else:
        trainer.train()


if __name__ == "__main__":
    typer.run(train)
