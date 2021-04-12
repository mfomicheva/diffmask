import os
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationBinaryClassification, QualityEstimationRegression
from diffmask.options import make_parser


if __name__ == '__main__':
    parser = make_parser()
    hparams = parser.parse_args()
    print(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=hparams.model_path,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
    )

    assert not os.path.exists(hparams.model_path)
    if hparams.num_labels == 1:
        qe = QualityEstimationRegression(hparams)
    elif hparams.num_labels == 2:
        qe = QualityEstimationBinaryClassification(hparams)
    else:
        raise NotImplementedError

    trainer = pl.Trainer(
        gpus=int(hparams.use_cuda),
        progress_bar_refresh_rate=0,
        max_epochs=hparams.epochs,
        check_val_every_n_epoch=1,
        logger=pl.loggers.TensorBoardLogger("outputs", name="qe"),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(qe)