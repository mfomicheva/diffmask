import unittest
import os
import tempfile

import pytorch_lightning as pl
from transformers import XLMRobertaTokenizer

from diffmask.models.quality_estimation import load_sent_level
from diffmask.models.quality_estimation import QualityEstimationRegression
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationRoberta
from diffmask.models.quality_estimation import QualityEstimationBinaryClassificationBert
from diffmask.options import make_parser

from tests.util import create_dummy_data


class TestQualityEstimationModel(unittest.TestCase):

    def test_prepare_data(self):
        bert_model = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(bert_model)
        file_prefix = 'test'
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            create_dummy_data(data_dir, file_prefix)
            dataset = load_sent_level(
                os.path.join(data_dir, f'{file_prefix}.src'),
                os.path.join(data_dir, f'{file_prefix}.tgt'),
                os.path.join(data_dir, f'{file_prefix}.labels'),
                tokenizer,
                path_word_labels=os.path.join(data_dir, f'{file_prefix}.tags')
            )
            assert dataset[0][1][0].shape[0] == 128
            assert sum(dataset[0][1][1]).item() == 20
            assert dataset[0][1][2].item() == 0

    def _make_hparams(self, data_dir, architecture, pretrained_model_name, target_only=False):
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

    def _train_model(self, data_dir, hparams):
        create_dummy_data(data_dir, 'train', num_examples=50)
        create_dummy_data(data_dir, 'valid', num_examples=10)
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

    def test_train_model(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = self._make_hparams(data_dir, 'roberta', 'xlm-roberta-base')
            self._train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)

    def test_train_model_bert(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = self._make_hparams(data_dir, 'bert', 'bert-base-multilingual-cased')
            self._train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)

    def test_train_model_target_only(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = self._make_hparams(data_dir, 'roberta', 'xlm-roberta-base', target_only=True)
            self._train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
