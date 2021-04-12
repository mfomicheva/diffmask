import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store_true', default=False)
    parser.add_argument("--model", type=str, default="xlm-roberta-base", choices=["bert-base-multilingual-cased", "xlm-roberta-base"])
    parser.add_argument("--architecture", type=str, default="roberta", choices=["bert", "roberta"])
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
    parser.add_argument("--class_weighting", default=False, action='store_true')
    parser.add_argument("--val_loss", default="f1", choices=["f1", "mcc", "mse"])
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--target_only", action='store_true', default=False)
    return parser
