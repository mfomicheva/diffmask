import unittest
import os
import tempfile

from transformers import XLMRobertaTokenizer, BertTokenizer

from diffmask.models.quality_estimation import load_sent_level

from tests.util import create_dummy_data, make_hparams_train, train_model


class TestQualityEstimationModel(unittest.TestCase):

    def test_prepare_data_roberta(self):
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
                'roberta',
                path_word_labels=os.path.join(data_dir, f'{file_prefix}.tags')
            )
            assert dataset[0][1][0].shape[0] == 128
            assert sum(dataset[0][1][1]).item() == 20
            assert dataset[0][1][2].shape[0] == 128
            assert sum(dataset[0][1][2]).item() == 0
            assert dataset[0][1][3].item() == 0

    def test_prepare_data_bert(self):
        bert_model = "bert-base-multilingual-cased"
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        file_prefix = 'test'
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            create_dummy_data(data_dir, file_prefix)
            dataset = load_sent_level(
                os.path.join(data_dir, f'{file_prefix}.src'),
                os.path.join(data_dir, f'{file_prefix}.tgt'),
                os.path.join(data_dir, f'{file_prefix}.labels'),
                tokenizer,
                'bert',
                path_word_labels=os.path.join(data_dir, f'{file_prefix}.tags')
            )
            assert dataset[0][1][0].shape[0] == 128
            assert sum(dataset[0][1][1]).item() == 19
            assert dataset[0][1][2].shape[0] == 128
            assert sum(dataset[0][1][2]).item() == 2
            assert dataset[0][1][3].item() == 0

    def test_train_model(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base')
            train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)

    def test_train_model_regression(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base', num_labels=1, val_loss='pearson', )
            train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)

    def test_train_model_bert(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'bert', 'bert-base-multilingual-cased')
            train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)

    def test_train_model_target_only(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base', target_only=True)
            train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
