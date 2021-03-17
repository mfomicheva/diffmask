import os
import argparse
import pytorch_lightning as pl

from diffmask.models.quality_estimation import QualityEstimationClassification
from diffmask.utils.evaluate_qe import generate_predictions


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

    hparams = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu
    device = "cuda:{}".format(hparams.gpu)

    qe = QualityEstimationClassification.load_from_checkpoint(hparams.model_path).to(device)
    qe.hparams = hparams
    qe.freeze()
    qe.prepare_data()
    loader = qe.test_dataloader()
    predictions = generate_predictions(qe, loader, device, evaluate=True)
