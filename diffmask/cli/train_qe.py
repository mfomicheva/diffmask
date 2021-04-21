import torch
import os
import numpy as np
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.options import make_train_parser


if __name__ == '__main__':
    parser = make_train_parser()
    hparams = parser.parse_args()
    print(hparams)

    if hparams.seed is not None:
        torch.manual_seed(hparams.seed)
        np.random.seed(hparams.seed)

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
        if hparams.architecture == 'roberta':
            qe = QualityEstimationBinaryClassificationRoberta(hparams)
        elif hparams.architecture == 'bert':
            qe = QualityEstimationBinaryClassificationBert(hparams)
        else:
            raise ValueError
    else:
        raise NotImplementedError

    trainer = pl.Trainer(
        gpus=int(hparams.use_cuda),
        progress_bar_refresh_rate=0,
        max_epochs=hparams.epochs,
        check_val_every_n_epoch=1,
        logger=pl.loggers.TensorBoardLogger("outputs", name="qe"),
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=hparams.clip_grad,
    )
    trainer.fit(qe)
