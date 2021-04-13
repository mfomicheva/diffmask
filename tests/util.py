import torch
import random
import os

import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.options import make_parser


torch.manual_seed(1234)
random.seed(1234)

vocab = ['es', 'una', 'frase', 'con', 'muchas', 'palabras', 'this', 'is', 'a', 'sentence', 'with', 'many', 'words']


def _make_files(data, num_examples, max_len, data_dir, file_prefix):
    offset = 0
    with open(os.path.join(data_dir, f'{file_prefix}.src'), 'w') as s:
        with open(os.path.join(data_dir, f'{file_prefix}.tgt'), 'w') as t:
            with open(os.path.join(data_dir, f'{file_prefix}.labels'), 'w') as l:
                with open(os.path.join(data_dir, f'{file_prefix}.tags'), 'w') as w:
                    for _ in range(num_examples):
                        for fh in (s, t):
                            ex_len = random.randint(1, max_len + 1)
                            print(' '.join([vocab[i] for i in data[offset:offset + ex_len]]), file=fh)
                            offset += ex_len
                        print(torch.randint(2, (1,)).item(), file=l)
                        print(' '.join(map(str, torch.randint(2, (ex_len,)).tolist())), file=w)


def create_dummy_data(data_dir, file_prefix, num_examples=100, max_len=100):
    data = torch.randint(len(vocab), (num_examples * max_len * 2,))
    _make_files(data, num_examples, max_len, data_dir, file_prefix)


def make_hparams(data_dir, architecture, pretrained_model_name, target_only=False):
    parser = make_parser()
    parser.parse_known_args()
    input_args = [
        '--architecture', architecture,
        '--model', pretrained_model_name,
        '--src_train_filename', os.path.join(data_dir, 'train.src'),
        '--tgt_train_filename', os.path.join(data_dir, 'train.tgt'),
        '--labels_train_filename', os.path.join(data_dir, 'train.labels'),
        '--word_labels_train_filename', os.path.join(data_dir, 'train.tags'),
        '--src_val_filename', os.path.join(data_dir, 'valid.src'),
        '--tgt_val_filename', os.path.join(data_dir, 'valid.tgt'),
        '--labels_val_filename', os.path.join(data_dir, 'valid.labels'),
        '--word_labels_val_filename', os.path.join(data_dir, 'valid.tags'),
        '--model_path', os.path.join(data_dir),
        '--epochs', '1',
    ]
    if target_only:
        input_args.append('--target_only')
    return parser.parse_args(input_args)


def train_model(data_dir, hparams):
    create_dummy_data(data_dir, 'train', num_examples=10)
    create_dummy_data(data_dir, 'valid', num_examples=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
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
        logger=pl.loggers.TensorBoardLogger("outputs", name="qe"),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(qe)
    print(checkpoint_callback.format_checkpoint_name(1, {}))
    return qe
