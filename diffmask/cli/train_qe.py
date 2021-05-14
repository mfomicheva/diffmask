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
    
    if hparams.model_pref is not None:
        model_name = '{}.'.format(hparams.model_pref)
    else:
        model_name = '{}.'.format(hparams.model)
    best_val_score = None
    best_model_path = None

    _op = min if hparams.opt_mode == 'min' else max

    for n in range(1, hparams.nfolds + 1):
        if hparams.seed is not None:
            torch.manual_seed(hparams.seed * n)
            np.random.seed(hparams.seed * n)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            prefix=model_name,
            filepath=hparams.model_path,
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
        )

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
            logger=pl.loggers.TensorBoardLogger(os.path.join(hparams.model_path, 'tb-logs'), name=model_name),
            checkpoint_callback=checkpoint_callback,
            gradient_clip_val=hparams.clip_grad,
        )
        trainer.fit(qe)
        
        current_best_model = _op(
            checkpoint_callback.best_k_models,
            key=checkpoint_callback.best_k_models.get
        )
        current_best_value = checkpoint_callback.best_k_models[current_best_model]
        
        print('Random seed: {}'.format(hparams.seed * n))
        print('Current best value: {}'.format(current_best_value))
        print('Current best model: {}'.format(current_best_model))        

        if best_val_score is None:
            best_val_score = current_best_value
            best_model_path = current_best_model
        elif _op((current_best_value, best_val_score)) != current_best_value or current_best_value == best_val_score:
            print('Previous best value: {}'.format(best_val_score))
            print('Deleting model {}'.format(current_best_model))
            os.remove(current_best_model)
        else:
            best_val_score = current_best_value
            best_model_path = current_best_model
            print('New best model: {}'.format(current_best_model))
    print('SAVED BEST MODEL AT: {}'.format(best_model_path))

