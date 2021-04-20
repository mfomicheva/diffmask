import torch
import pickle

from diffmask.attributions.schulz import schulz_explainer, hidden_states_statistics, schulz_loss
from diffmask.attributions.guan import guan_explainer, guan_loss
from diffmask.utils.util import map_bpe_moses, map_bpe_moses_bert
from diffmask.utils.exceptions import TokenizationError


class SampleAttributions:

    def __init__(self, input_ids, attributions, mask, tokenizer, sent_label, sent_pred=None):
        self.input_ids = input_ids
        self.attributions = attributions
        self.mask = mask
        self.tokenizer = tokenizer
        self.sent_pred = sent_pred
        self.sent_label = sent_label
        self.attributions_source = dict()
        self.attributions_target = dict()
        self.attributions_special = dict()
        self.input_ids_source = None
        self.input_ids_target = None
        self.input_ids_special = None

    def format_attributions(self, normalize=False, invert=False, target_only=False):
        self.remove_masked()
        num_layers = self.attributions.shape[1]
        for layer_id in range(num_layers):
            attributions_id = self.select_by_layer(self.attributions, layer_id)
            if normalize:
                attributions_id = self.normalize_attributions(attributions_id)
            if invert:
                attributions_id = self.invert(attributions_id)
            if not target_only:
                ids, ats = self.select_source(self.input_ids, self.tokenizer, attributions_id)
                self.attributions_source[layer_id] = ats
                self.input_ids_source = ids
            ids, ats = self.select_target(self.input_ids, self.tokenizer, attributions_id, target_only=target_only)
            self.attributions_target[layer_id] = ats
            self.input_ids_target = ids
            ids, ats = self.select_special(self.input_ids, self.tokenizer, attributions_id)
            self.attributions_special[layer_id] = ats
            self.input_ids_special = ids

    def remove_masked(self):
        self.input_ids = torch.as_tensor(self.input_ids)[:self.mask.sum(-1).item()].tolist()
        self.attributions = self.attributions[:self.mask.sum(-1).item()].cpu()

    @staticmethod
    def select_by_layer(attributions, layer_id):
        if layer_id == -1:
            return torch.mean(attributions, dim=-1)
        else:
            return attributions[:, layer_id]

    @staticmethod
    def normalize_attributions(attributions):
        return attributions / attributions.abs().max(0, keepdim=True).values

    @staticmethod
    def invert(attributions):
        return 1. - attributions

    @staticmethod
    def select_source(input_ids, tokenizer, attributions, with_special=False):
        start_id = 1
        end_id = input_ids.index(tokenizer.sep_token_id)
        if with_special:
            start_id -= 1
            end_id += 1
        return input_ids[start_id:end_id], attributions[start_id:end_id]

    @staticmethod
    def select_target(input_ids, tokenizer, attributions, target_only=False, with_special=False):
        t = torch.as_tensor(input_ids)
        cond = torch.where(t == tokenizer.sep_token_id, torch.ones(t.shape), torch.zeros(t.shape))
        eos_ids = torch.nonzero(cond).squeeze(dim=-1).tolist()
        if target_only:
            start_id = 1
        else:
            if len(eos_ids) == 3:
                start_id = eos_ids[1] + 1
            elif len(eos_ids) == 2:
                start_id = eos_ids[0] + 1
            else:
                raise ValueError
        end_id = eos_ids[-1]
        if with_special:
            start_id -= 1
            end_id += 1
        return input_ids[start_id:end_id], attributions[start_id:end_id]

    @staticmethod
    def select_special(input_ids, tokenizer, attributions):
        idx = input_ids.index(tokenizer.cls_token_id)
        return [idx], attributions[idx]


class SampleAttributionsMapping:

    def __init__(
            self, input_ids, attributions, moses_tokens, word_labels, sent_label, bpe_tokenizer, token_mapper_fn,
            sent_pred=None
    ):
        self.input_ids = input_ids
        self.attributions = attributions
        self.moses_tokens = moses_tokens
        self.tokenizer = bpe_tokenizer
        self.mapper_fn = token_mapper_fn
        self.sent_label = sent_label
        self.sent_pred = sent_pred
        self.word_labels = word_labels
        self.attributions_mapped = dict()
        self.attributions_bad = dict()

    def format_attributions(self):
        try:
            bpe_moses, moses_bpe = self.map_tokens(self.moses_tokens, self.input_ids, self.tokenizer, self.mapper_fn)
        except TokenizationError:
            raise TokenizationError
        for layer_id in self.attributions:
            self.attributions_mapped[layer_id] = self.map_attributions(moses_bpe, self.attributions[layer_id])
            self.attributions_bad[layer_id] = self.select_by_word_label(self.attributions_mapped[layer_id], self.word_labels, bpe_moses)

    @staticmethod
    def map_tokens(moses_tokens, input_ids, bpe_tokenizer, token_mapper_fn):
        bpe_tokens = bpe_tokenizer.convert_ids_to_tokens(input_ids)
        try:
            bpe_moses, moses_bpe = token_mapper_fn(bpe_tokens, moses_tokens)
        except TokenizationError:
            raise TokenizationError
        return bpe_moses, moses_bpe

    @staticmethod
    def map_attributions(moses_to_bpe, attributions):
        token_indexes = sorted(moses_to_bpe.keys())
        return [max([attributions[bpe_id] for bpe_id in moses_to_bpe[index]]) for index in token_indexes]

    @staticmethod
    def select_by_word_label(attributions, word_labels, bpe_moses_mapping):
        ids = [i for i, label in enumerate(word_labels) if label == 1]
        return [a for i, a in enumerate(attributions) if bpe_moses_mapping[i] in ids]


class AttributeQE:

    def __init__(self, model, getter, setter, layer_indexes, device, split='valid', guan=False, batch_size=None):
        self.model = model
        self.getter = getter
        self.setter = setter
        self.layer_indexes = layer_indexes
        self.split = split
        self.device = device

        self.text_dataset = self.model.test_dataset_orig if split == 'test' else self.model.val_dataset_orig
        self.dataset = self.model.test_dataset if split == 'test' else self.model.val_dataset
        self.attributions = None
        self.explainer_fn = guan_explainer if guan else schulz_explainer
        self.explainer_loss = guan_loss if guan else schulz_loss

        self.batch_size = batch_size if batch_size is not None else self.model.hparams.batch_size

    def make_attributions(self, verbose=False, save=None, load=None, input_only=True, steps=50):

        if load is not None:
            self.attributions = pickle.load(open(load, 'rb'))
            return
        if self.model.hparams.architecture == 'roberta':
            pretrained_model = self.model.net.roberta
        elif self.model.hparams.architecture == 'bert':
            pretrained_model = self.model.net.bert
        else:
            raise NotImplementedError
        all_q_z_loc, all_q_z_scale = hidden_states_statistics(self.model, pretrained_model, self.getter, input_only=input_only)
        result = []
        for batch_idx, sample in enumerate(torch.utils.data.DataLoader(self.dataset, batch_size=self.model.hparams.batch_size, num_workers=20)):
            input_ids, mask, _, labels = sample
            inputs_dict = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': mask.to(self.device),
                'labels': labels.to(self.device),
            }
            all_attributions = []
            for layer_idx in self.layer_indexes:
                all_q_z_idx = 0 if input_only else layer_idx
                kwargs = self.explainer_loss(
                    q_z_loc=all_q_z_loc[all_q_z_idx].unsqueeze(0).to(self.device),
                    q_z_scale=all_q_z_scale[all_q_z_idx].unsqueeze(0).to(self.device),
                    verbose=verbose,
                )
                layer_attributions = self.explainer_fn(
                    self.model.net,
                    inputs_dict=inputs_dict,
                    getter=self.getter,
                    setter=self.setter,
                    hidden_state_idx=layer_idx,
                    steps=steps,
                    lr=1e-1,
                    la=10,
                    **kwargs,
                )
                all_attributions.append(layer_attributions.unsqueeze(-1))
            all_attributions = torch.cat(all_attributions, -1)  # B, T, L
            for bidx in range(all_attributions.shape[0]):
                try:
                    result.append(all_attributions[bidx, :, :])  # T, L
                except IndexError:
                    break
        if save is not None:
            pickle.dump(result, open(save, 'wb'))
        self.attributions = result

    def make_data(self, silent=False, normalize=False, invert=False, target_only=False, predictions=None):
        res = []
        tokens_mapper_fn = map_bpe_moses if self.model.hparams.architecture == 'roberta' else map_bpe_moses_bert
        for sentid in range(len(self.attributions)):
            source_tokens = self.text_dataset[sentid][0].split()
            target_tokens = self.text_dataset[sentid][1].split()
            word_labels = self.text_dataset[sentid][3]
            sent_pred = predictions[sentid] if predictions is not None else None
            if len(source_tokens) == 0 or len(target_tokens) == 0:
                if not silent:
                    print('Empty segment! Skipping...')
                continue
            try:
                sample_attributions, mapped_source, mapped_target = self.format_sample_attributions(
                    target_tokens, word_labels, self.dataset[sentid], self.attributions[sentid], self.model.tokenizer,
                    tokens_mapper_fn, normalize=normalize, invert=invert, target_only=target_only,
                    source_tokens=source_tokens, sent_pred=sent_pred,
                )
            except TokenizationError:
                if not silent:
                    print('BPE mapping error! Skipping...')
                continue
            try:
                assert len(word_labels) == len(mapped_target.attributions_mapped[0])
            except AssertionError:
                if not silent:
                    print('Sequence too long. Skipping')
                continue
            res.append((sample_attributions, mapped_source, mapped_target,))
        return res

    @staticmethod
    def format_sample_attributions(
            tokens, word_labels, sample, attributions, tokenizer, token_mapper_fn, normalize=False, invert=False,
            target_only=False, source_tokens=None, sent_pred=None
    ):
        input_ids, mask, _, sent_labels = sample
        sample_attributions = SampleAttributions(
            input_ids, attributions, mask, tokenizer, sent_labels.item(), sent_pred=sent_pred)
        sample_attributions.format_attributions(normalize=normalize, invert=invert, target_only=target_only)
        mapped_source = None
        if not target_only:
            mapped_source = SampleAttributionsMapping(
                sample_attributions.input_ids_source,
                sample_attributions.attributions_source,
                source_tokens, word_labels, sent_labels.item(), tokenizer, token_mapper_fn, sent_pred=sent_pred
            )
            try:
                mapped_source.format_attributions()
            except TokenizationError:
                raise
        mapped_target = SampleAttributionsMapping(
            sample_attributions.input_ids_target,
            sample_attributions.attributions_target,
            tokens, word_labels, sent_labels.item(), tokenizer, token_mapper_fn, sent_pred=sent_pred
        )
        try:
            mapped_target.format_attributions()
        except TokenizationError:
            raise
        return sample_attributions, mapped_source, mapped_target
