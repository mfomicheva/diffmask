import torch
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
    get_constant_schedule_with_warmup,
)

from ..utils.metrics import accuracy_precision_recall_f1, matthews_corr_coef


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


def load_sent_level(path_src, path_tgt, path_labels, tokenizer, max_seq_length=128, path_word_labels=None):

    def _read_file(path, label=False):
        return [float(l.strip()) if label else l.strip() for l in open(path)]

    def _read_word_labels(path):
        out = []
        for l in open(path):
            out.append([int(val.strip()) for val in l.split()])
        return out

    srcs = _read_file(path_src)
    tgts = _read_file(path_tgt)
    labels = _read_file(path_labels, label=True)
    assert len(srcs) == len(tgts) == len(labels)
    word_labels = None
    if path_word_labels is not None:
        word_labels = _read_word_labels(path_word_labels)
        assert len(word_labels) == len(srcs)
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
        word_labels_i = word_labels[i] if word_labels else None
        data_text.append((srcs[i], tgts[i], labels[i], word_labels_i))

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
        self.train_dataset = None
        self.train_dataset_orig = None
        self.val_dataset = None
        self.val_dataset_orig = None
        self.test_dataset = None
        self.test_dataset_orig = None

    def prepare_data(self):
        if self.hparams.src_train_filename is not None:
            self.train_dataset, self.train_dataset_orig = load_sent_level(
                self.hparams.src_train_filename, self.hparams.tgt_train_filename, self.hparams.labels_train_filename,
                self.tokenizer, path_word_labels=self.hparams.word_labels_train_filename,
            )
        if self.hparams.src_val_filename is not None:
            self.val_dataset, self.val_dataset_orig = load_sent_level(
                self.hparams.src_val_filename, self.hparams.tgt_val_filename, self.hparams.labels_val_filename,
                self.tokenizer, path_word_labels=self.hparams.word_labels_val_filename
            )
        if self.hparams.src_test_filename is not None:
            self.test_dataset, self.test_dataset_orig = load_sent_level(
                self.hparams.src_test_filename, self.hparams.tgt_test_filename, self.hparams.labels_test_filename,
                self.tokenizer, path_word_labels=self.hparams.word_labels_test_filename
            )

    def make_sampler(self):
        labels = [t[-1] for t in self.train_dataset]
        labels_tensor = torch.LongTensor(labels)
        class_sample_count = torch.bincount(labels_tensor)
        print(class_sample_count)
        weights = 1. / class_sample_count.to(torch.double)
        instance_weights = []
        for l in labels:
            instance_weights.append(weights[l])
        sampler = WeightedRandomSampler(instance_weights, len(instance_weights))
        return sampler

    def train_dataloader(self):
        sampler = self.make_sampler() if self.hparams.class_weighting else None
        shuffle = False if sampler else True
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, sampler=sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size
        )

    def training_step(self, batch, batch_idx=None):
        input_ids, mask, _, labels = batch

        logits = self.forward(input_ids, mask)[0]
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(-1)
        acc, _, _, f1 = accuracy_precision_recall_f1(logits.argmax(-1), labels, average=True)

        outputs_dict = {
            "acc": acc,
            "f1": f1,
        }

        if self.hparams.val_loss == "mcc":
            mcc = matthews_corr_coef(logits.argmax(-1), labels)
            outputs_dict.update({"mcc": mcc})

        outputs_dict = {
            "loss": loss,
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        return outputs_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        val_metrics = ["val_acc", "val_f1"]
        if self.hparams.val_loss not in val_metrics:
            val_metrics.append("val_{}".format(self.hparams.val_loss))

        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in val_metrics
        }

        outputs_dict = {
            "val_loss": -outputs_dict["val_{}".format(self.hparams.val_loss)],
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class QualityEstimationBinaryClassification(QualityEstimation):

    def __init__(self, hparams):
        super().__init__(hparams)

        config = XLMRobertaConfig.from_pretrained(self.hparams.model)
        config.num_labels = 2
        self.net = XLMRobertaForSequenceClassification.from_pretrained(self.hparams.model, config=config)

    def forward(self, input_ids, mask, labels=None):
        return self.net(input_ids=input_ids, attention_mask=mask)
