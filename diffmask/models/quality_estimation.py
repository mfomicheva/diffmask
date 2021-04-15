import torch
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl

from scipy.stats import pearsonr

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from ..utils.metrics import accuracy_precision_recall_f1, matthews_corr_coef


def load_sent_level(
        path_src, path_tgt, path_labels, tokenizer, architecture, regression=False, max_seq_length=128,
        path_word_labels=None, target_only=False,
):

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
    data = []
    data_text = []
    for i in range(len(srcs)):
        if target_only:
            text_a = tgts[i]
            text_b = None
        else:
            text_a = srcs[i]
            text_b = tgts[i]
        result = tokenizer.encode_plus(
            text_a, text_pair=text_b, padding='max_length', truncation='longest_first', max_length=max_seq_length,
            pad_to_max_length=True,
        )
        if architecture == 'roberta':
            result['token_type_ids'] = torch.zeros((len(result['input_ids']),)).tolist()
        data.append(result)
        word_labels_i = word_labels[i] if word_labels else None
        data_text.append((srcs[i], tgts[i], labels[i], word_labels_i))

    tensor_dataset = [
        torch.tensor([d['input_ids'] for d in data], dtype=torch.long),
        torch.tensor([d['attention_mask'] for d in data], dtype=torch.long),
        torch.tensor([d['token_type_ids'] for d in data], dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32 if regression else torch.long),
    ]
    return torch.utils.data.TensorDataset(*tensor_dataset), data_text


class QualityEstimation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.train_dataset_orig = None
        self.val_dataset = None
        self.val_dataset_orig = None
        self.test_dataset = None
        self.test_dataset_orig = None
        self.regression = False
        self.tokenizer = None

    def prepare_data(self):
        if self.hparams.src_train_filename is not None:
            self.train_dataset, self.train_dataset_orig = load_sent_level(
                self.hparams.src_train_filename, self.hparams.tgt_train_filename, self.hparams.labels_train_filename,
                self.tokenizer, self.hparams.architecture, path_word_labels=self.hparams.word_labels_train_filename,
                regression=self.regression, target_only=self.hparams.target_only
            )
        if self.hparams.src_val_filename is not None:
            self.val_dataset, self.val_dataset_orig = load_sent_level(
                self.hparams.src_val_filename, self.hparams.tgt_val_filename, self.hparams.labels_val_filename,
                self.tokenizer, self.hparams.architecture, path_word_labels=self.hparams.word_labels_val_filename,
                regression=self.regression, target_only=self.hparams.target_only
            )
        if self.hparams.src_test_filename is not None:
            self.test_dataset, self.test_dataset_orig = load_sent_level(
                self.hparams.src_test_filename, self.hparams.tgt_test_filename, self.hparams.labels_test_filename,
                self.tokenizer, self.hparams.architecture, path_word_labels=self.hparams.word_labels_test_filename,
                regression=self.regression, target_only=self.hparams.target_only
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
        loss = self.loss(logits, labels)
        outputs_dict = self.compute_metrics(logits, labels, loss)

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

        val_metrics = ["val_{}".format(m) for m in self.metrics]
        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in val_metrics
        }

        outputs_dict = {
            "val_loss": self.val_loss(outputs_dict["val_{}".format(self.hparams.val_loss)]),
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        import math
        total = len(self.train_dataloader()) // self.hparams.epochs
        warmup_steps = math.ceil(total * 0.06)
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total
        )
        optimizers = [optimizer]
        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "step",
            },
        ]
        return optimizers, schedulers

    def forward(self, input_ids, mask, labels=None):
        return self.net(input_ids=input_ids, attention_mask=mask, labels=labels)


class QualityEstimationBinaryClassification(QualityEstimation):

    @staticmethod
    def val_loss(val_loss):
        return -val_loss

    @staticmethod
    def loss(logits, labels):
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(-1)
        return loss

    @staticmethod
    def compute_metrics(logits, labels, loss):
        acc, _, _, f1 = accuracy_precision_recall_f1(logits.argmax(-1), labels, average=True)
        mcc = matthews_corr_coef(logits.argmax(-1), labels)
        outputs_dict = {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
        }
        return outputs_dict


class QualityEstimationBinaryClassificationRoberta(QualityEstimationBinaryClassification):

    def __init__(self, hparams):
        super().__init__(hparams)

        config = XLMRobertaConfig.from_pretrained(self.hparams.model)
        config.num_labels = 2
        self.net = XLMRobertaForSequenceClassification.from_pretrained(self.hparams.model, config=config)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.hparams.model)
        self.metrics = ["f1", "acc", "mcc"]
        if self.hparams.val_loss not in self.metrics:
            self.metrics.append(self.hparams.val_loss)


class QualityEstimationBinaryClassificationBert(QualityEstimationBinaryClassification):

    def __init__(self, hparams):
        super().__init__(hparams)

        config = BertConfig.from_pretrained(self.hparams.model)
        config.num_labels = 2
        self.net = BertForSequenceClassification.from_pretrained(self.hparams.model, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model)
        self.metrics = ["f1", "acc", "mcc"]
        if self.hparams.val_loss not in self.metrics:
            self.metrics.append(self.hparams.val_loss)


class QualityEstimationRegression(QualityEstimation):

    def __init__(self, hparams):
        super().__init__(hparams)

        config = XLMRobertaConfig.from_pretrained(self.hparams.model)
        config.num_labels = 1
        self.net = XLMRobertaForSequenceClassification.from_pretrained(self.hparams.model, config=config)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.hparams.model)
        self.metrics = ["mse", "pearson"]
        self.regression = True

    @staticmethod
    def val_loss(val_loss):
        return val_loss

    @staticmethod
    def loss(logits, labels):
        loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 1), reduction="mean")
        return loss

    @staticmethod
    def compute_metrics(logits, labels, loss):
        yhat = logits.squeeze().detach().cpu().numpy()
        y = labels.squeeze().detach().cpu().numpy()
        return {"mse": loss, "pearson": -1 * pearsonr(y, yhat)[0]}
