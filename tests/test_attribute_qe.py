import os
import unittest
import tempfile

from diffmask.attributions.schulz import qe_roberta_schulz_explainer
from diffmask.attributions.guan import qe_roberta_guan_explainer
from diffmask.attributions.integrated_gradient import qe_integrated_gradient_explainer
from diffmask.attributions.attention import qe_roberta_attention_explainer
from diffmask.utils.evaluate_qe import EvaluateQE
from diffmask.attributions.attribute_qe import make_data

from tests.util import make_hparams_train, train_model


class TestAttributeQE(unittest.TestCase):

    def test_attribute_and_evaluate_qe_roberta(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base')
            qe = train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
            tensor_dataset = qe.val_dataset
            text_dataset = qe.val_dataset_orig
            attributions = qe_roberta_schulz_explainer(
                qe, tensor_dataset, save=None, input_only=False, steps=1, batch_size=2,
                num_layers=14, learning_rate=1e-1, aux_loss_weight=10, num_workers=1
            )
            data = make_data(tensor_dataset, text_dataset, qe, attributions)
            eval_cls = EvaluateQE()
            score = eval_cls.auc_score(data, 0)
            acc = eval_cls.top1_accuracy(data, 0)
            import numpy as np
            eval_cls.attributions_types(data, 0, np.mean)

    def test_attribute_and_evaluate_qe_roberta_guan(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base')
            qe = train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
            tensor_dataset = qe.val_dataset
            text_dataset = qe.val_dataset_orig
            attributions = qe_roberta_guan_explainer(
                qe, tensor_dataset, save=None, steps=1, batch_size=2,
                num_layers=14, learning_rate=1e-1, aux_loss_weight=10
            )
            data = make_data(tensor_dataset, text_dataset, qe, attributions)
            assert len(data) > 0

    def test_attribute_and_evaluate_qe_roberta_gradients(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base')
            qe = train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
            tensor_dataset = qe.val_dataset
            text_dataset = qe.val_dataset_orig
            attributions = qe_integrated_gradient_explainer(
                qe, tensor_dataset, save=None, steps=1, batch_size=2,
                num_layers=14, learning_rate=1e-1, aux_loss_weight=10
            )
            data = make_data(tensor_dataset, text_dataset, qe, attributions)
            assert len(data) > 0

    def test_attribute_and_evaluate_qe_roberta_attention(self):
        with tempfile.TemporaryDirectory("test_prepare_data") as data_dir:
            hparams = make_hparams_train(data_dir, 'roberta', 'xlm-roberta-base')
            qe = train_model(data_dir, hparams)
            assert 'epoch=0.ckpt' in os.listdir(data_dir)
            tensor_dataset = qe.val_dataset
            text_dataset = qe.val_dataset_orig
            attributions = qe_roberta_attention_explainer(qe, tensor_dataset, num_workers=1)
            data = make_data(tensor_dataset, text_dataset, qe, attributions)
            assert len(data) > 0
