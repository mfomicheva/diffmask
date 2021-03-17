import os
import argparse
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationClassification


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--src_train_filename", type=str)
    parser.add_argument("--tgt_train_filename", type=str)
    parser.add_argument("--labels_train_filename", type=str)
    parser.add_argument("--word_labels_train_filename", type=str, default=None)
    parser.add_argument("--src_val_filename", type=str)
    parser.add_argument("--tgt_val_filename", type=str)
    parser.add_argument("--labels_val_filename", type=str)
    parser.add_argument("--word_labels_val_filename", type=str, default=None)
    parser.add_argument("--src_test_filename", type=str)
    parser.add_argument("--tgt_test_filename", type=str)
    parser.add_argument("--labels_test_filename", type=str)
    parser.add_argument("--word_labels_test_filename", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=10)

    hparams = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=hparams.model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
    )

    assert not os.path.exists(hparams.model_path)
    qe = QualityEstimationClassification(hparams)
    trainer = pl.Trainer(
        gpus=int(hparams.gpu != ""),
        progress_bar_refresh_rate=0,
        max_epochs=hparams.epochs,
        check_val_every_n_epoch=1,
        logger=pl.loggers.TensorBoardLogger("outputs", name="qe"),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(qe)
