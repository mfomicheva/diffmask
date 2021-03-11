import os
import argparse
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationClassification


if __name__ == '__main__':
    mlqe_parser = argparse.ArgumentParser()
    mlqe_parser.add_argument("--gpu", type=str, default="0")
    mlqe_parser.add_argument("--model", type=str, default="xlm-roberta-base")
    mlqe_parser.add_argument("--src_train_filename", type=str, default="./datasets/qe/mlqe-multilingual/train.all.src")
    mlqe_parser.add_argument("--tgt_train_filename", type=str, default="./datasets/qe/mlqe-multilingual/train.all.mt")
    mlqe_parser.add_argument("--labels_train_filename", type=str, default="./datasets/qe/mlqe-multilingual/train.all.DA-bin")
    mlqe_parser.add_argument("--word_labels_train_filename", type=str, default=None)
    mlqe_parser.add_argument("--src_val_filename", type=str, default="./datasets/qe/mlqe-multilingual/dev.all.src")
    mlqe_parser.add_argument("--tgt_val_filename", type=str, default="./datasets/qe/mlqe-multilingual/dev.all.mt")
    mlqe_parser.add_argument("--labels_val_filename", type=str, default="./datasets/qe/mlqe-multilingual/dev.all.DA-bin")
    mlqe_parser.add_argument("--word_labels_val_filename", type=str, default=None)
    mlqe_parser.add_argument("--src_test_filename", type=str, default="./datasets/qe/en-et/test.src")
    mlqe_parser.add_argument("--tgt_test_filename", type=str, default="./datasets/qe/en-et/test.tgt")
    mlqe_parser.add_argument("--labels_test_filename", type=str, default="./datasets/qe/en-et/test.sent-labels")
    mlqe_parser.add_argument("--word_labels_test_filename", type=str, default="./datasets/qe/en-et/test.ignore-minor.word-labels")
    mlqe_parser.add_argument("--batch_size", type=int, default=32)
    mlqe_parser.add_argument("--seed", type=float, default=0)
    mlqe_parser.add_argument("--model_path", type=str, default="./models/qe_mlqe.ckpt")
    mlqe_parser.add_argument("--learning_rate", type=float, default=3e-5)
    mlqe_parser.add_argument("--epochs", type=int, default=10)

    parser = mlqe_parser
    hparams, _ = parser.parse_known_args()
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