import torch
import pytorch_lightning as pl
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_sent_level(path_src, path_tgt, path_labels, tokenizer, max_seq_length=128):
    def _read_file(path, label=False):
        return [float(l.strip()) if label else l.strip() for l in open(path)]
    srcs = _read_file(path_src)
    tgts = _read_file(path_tgt)
    labels = _read_file(path_labels, label=True)
    assert len(srcs) == len(tgts) == len(labels)
    data_tuples = []
    data_text = []
    for i in range(len(srcs)):
        src = tokenizer.tokenize(srcs[i])
        tgt = tokenizer.tokenize(tgts[i])
        _truncate_seq_pair(src, tgt, max_seq_length - 4)  # 4 is for special tokens in XLMRoberta
        tokens = src + [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        tokens += [tokenizer.sep_token]
        segment_ids += [1]
        tokens += tgt + [tokenizer.sep_token]
        segment_ids += [1] * (len(tgt) + 1)
        tokens = [tokenizer.cls_token] + tokens
        segment_ids = [0] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        data_tuples.append((input_ids, input_mask, segment_ids, labels[i]))
        data_text.append((srcs[i], tgts[i], labels[i]))

    tensor_dataset = [
        torch.tensor([d[0] for d in data_tuples], dtype=torch.long),
        torch.tensor([d[1] for d in data_tuples], dtype=torch.long),
        torch.tensor([d[2] for d in data_tuples], dtype=torch.long),
        torch.tensor([d[3] for d in data_tuples], dtype=torch.long),
    ]
    return torch.utils.data.TensorDataset(*tensor_dataset), data_text


class QualityEstimation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.hparams.model)

    def prepare_data(self):
        self.train_dataset, self.train_dataset_orig = load_sent_level(
            self.hparams.src_train_filename, self.hparams.tgt_train_filename, self.hparams.labels_train_filename,
            self.tokenizer,
        )
        self.val_dataset, self.val_dataset_orig = load_sent_level(
            self.hparams.src_val_filename, self.hparams.tgt_val_filename, self.hparams.labels_val_filename,
            self.tokenizer,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size
        )
