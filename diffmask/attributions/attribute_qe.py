import torch
import pickle

from diffmask.attributions.schulz import schulz_explainer, roberta_hidden_states_statistics, schulz_loss
from diffmask.attributions.guan import guan_explainer, guan_loss
from diffmask.utils.util import map_bpe_moses

# TODO: map to source/target/special at the data preparation stage


class SampleAttributions:

    def __init__(
            self, source_tokens, target_tokens, bpe_tokens, bpe_attributions, word_labels, sent_label,
            sent_pred, layer_id, normalize=False, invert=False,
    ):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.bpe_tokens = bpe_tokens
        self.bpe_attributions = bpe_attributions
        self.word_labels = word_labels
        self.sent_label = sent_label
        self.sent_pred = sent_pred
        self.layer_id = layer_id

        self.cls_idx = 0
        self.sep_idx = self.bpe_tokens.index('</s>')
        self.eos_idx = len(self.bpe_tokens) - 1

        self.normalize = normalize
        self.invert = invert

        self.bpe_attributions_layer = None
        self.source_token_attributions = None
        self.target_token_attributions = None
        self.special_token_attributions = None
        self.error_token_attributions = None
        self.random_attributions = None
        self.bpe_moses_target = None

    def map_attributions(self):
        try:
            src_bpe_moses, src_moses_bpe = map_bpe_moses(self.source_bpe_tokens(), self.source_tokens)
            tgt_bpe_moses, tgt_moses_bpe = map_bpe_moses(self.target_bpe_tokens(), self.target_tokens)
        except ValueError:
            raise
        self.set_layer_bpe_attributions()
        self.source_token_attributions = self._max_attribution(src_moses_bpe, self.source_bpe_attributions())
        self.target_token_attributions = self._max_attribution(tgt_moses_bpe, self.target_bpe_attributions())
        self.special_token_attributions = torch.tensor([self.bpe_attributions_layer[idx] for idx in (
            self.cls_idx, self.sep_idx, self.sep_idx + 1, self.eos_idx)])
        self.error_token_attributions = torch.tensor(self._error_attributions(
            self.target_bpe_attributions(), self.word_labels, tgt_bpe_moses))

    def set_layer_bpe_attributions(self):
        if self.layer_id == -1:
            self.bpe_attributions_layer = torch.mean(self.bpe_attributions, dim=-1)
        else:
            self.bpe_attributions_layer = self.bpe_attributions[:, self.layer_id]
        if self.normalize:
            self.bpe_attributions_layer = self.bpe_attributions_layer / self.bpe_attributions_layer.abs().max(0, keepdim=True).values
        if self.invert:
            self.bpe_attributions_layer = 1. - self.bpe_attributions_layer

    def source_bpe_tokens(self):
        return self.bpe_tokens[1:self.sep_idx]

    def target_bpe_tokens(self):
        return self.bpe_tokens[self.sep_idx+2: self.eos_idx]

    def source_bpe_attributions(self):
        return self.bpe_attributions_layer[1:self.sep_idx]

    def target_bpe_attributions(self):
        return self.bpe_attributions_layer[self.sep_idx+2:self.eos_idx]

    @staticmethod
    def _error_attributions(attributions, labels, mapping):
        error_idxs = [i for i, l in enumerate(labels) if l == 1]
        return [a for i, a in enumerate(attributions) if mapping[i] in error_idxs]

    @staticmethod
    def _max_attribution(moses_to_bpe, bpe_attributions):
        token_indexes = sorted(moses_to_bpe.keys())
        return [max([bpe_attributions[bpe_idx] for bpe_idx in moses_to_bpe[index]]) for index in token_indexes]


class AttributeQE:

    def __init__(self, model, getter, setter, layer_indexes, device, split='valid', guan=False):
        self.model = model
        self.getter = getter
        self.setter = setter
        self.layer_indexes = layer_indexes
        self.split = split
        self.device = device

        self.loader = self.model.test_dataloader() if split == 'test' else self.model.val_dataloader()
        self.text_dataset = self.model.test_dataset_orig if split == 'test' else self.model.val_dataset_orig
        self.dataset = self.model.test_dataset if split == 'test' else self.model.val_dataset
        self.attributions = None
        self.explainer_fn = guan_explainer if guan else schulz_explainer
        self.explainer_loss = guan_loss if guan else schulz_loss

    def make_attributions(self, verbose=False, save=None, load=None, input_only=True):

        if load is not None:
            self.attributions = pickle.load(open(load, 'rb'))
            return

        all_q_z_loc, all_q_z_scale = roberta_hidden_states_statistics(self.model, input_only=input_only)
        result = []
        for batch_idx, sample in enumerate(self.loader):
            input_ids, mask, labels = sample
            inputs_dict = {
                'input_ids': input_ids.to(self.device),
                'attention_mask': mask.to(self.device),
                'labels': labels.to(self.device),
            }
            all_attributions = []
            for layer_idx in self.layer_indexes:
                kwargs = self.explainer_loss(
                    q_z_loc=all_q_z_loc[layer_idx].unsqueeze(0).to(self.device),
                    q_z_scale=all_q_z_scale[layer_idx].unsqueeze(0).to(self.device),
                    verbose=verbose,
                )
                layer_attributions = self.explainer_fn(
                    self.model.net,
                    inputs_dict=inputs_dict,
                    getter=self.getter,
                    setter=self.setter,
                    hidden_state_idx=layer_idx,
                    steps=10,
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

    def make_data(self, layer_id, predictions=None, silent=False, normalize=False, invert=False):
        if layer_id != -1:
            assert layer_id in self.layer_indexes
            layer_id = self.layer_indexes.index(layer_id)
        res = []
        for sentid in range(len(self.attributions)):
            input_ids, mask, sent_labels = self.dataset[sentid]
            bpe_tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids.squeeze()[:mask.sum(-1).item()].squeeze())
            sent_pred = predictions[sentid] if predictions is not None else None
            bpe_attributions = self.attributions[sentid][:mask.sum(-1).item()].cpu()
            sample = SampleAttributions(
                self.text_dataset[sentid][0].split(), self.text_dataset[sentid][1].split(), bpe_tokens,
                bpe_attributions, self.text_dataset[sentid][3], sent_labels.item(), sent_pred, layer_id,
                normalize=normalize, invert=invert
            )
            if len(sample.source_tokens) == 0 or len(sample.target_tokens) == 0 or len(sample.bpe_tokens) == 0:
                if not silent:
                    print('Empty segment! Skipping...')
                continue
            try:
                sample.map_attributions()
            except ValueError:
                if not silent:
                    print('BPE mapping error! Skipping...')
                continue
            try:
                assert len(sample.word_labels) == len(sample.target_token_attributions)
            except AssertionError:
                if not silent:
                    print('Sequence too long. Skipping')
                continue
            res.append(sample)
        return res
