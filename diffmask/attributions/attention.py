import torch
import pickle


def qe_roberta_attention_explainer(
        qe_model, tensor_dataset, save=None, batch_size=1, num_workers=20, head_idx=-1, load=None, steps=50,
        num_layers=14, learning_rate=1e-1, aux_loss_weight=10, verbose=False, input_only=False,
):
    device = next(qe_model.parameters()).device
    qe_model.net.roberta.encoder.output_attentions = True
    for l in qe_model.net.roberta.encoder.layer:
        l.attention.self.output_attentions = True
    assert batch_size == 1
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, num_workers=num_workers)
    attributions_all_batched = []
    for batch_idx, sample in enumerate(loader):
        input_ids, mask, _, labels = sample
        inputs_dict = {
            'input_ids': input_ids.to(device),
            'mask': mask.to(device),
            'labels': labels.to(device),
        }
        outputs = qe_model(**inputs_dict)
        attention = outputs[-1]

        if head_idx != -1:
            attributions_att = torch.stack(
                [e.max(1).values.mean(-2) for e in attention], -1  # we take the maximum value from the heads
            )
        else:
            attributions_att = torch.stack(
                [e[:, head_idx, :, :].mean(-2) for e in attention], -1
            )
        attributions_att = torch.cat(
            (torch.full_like(attributions_att[..., 0:1], float("nan")),
             attributions_att,
             torch.full_like(attributions_att[..., 0:1], float("nan"))), -1
        )
        attributions_all_batched.append(attributions_att.squeeze())

    if save is not None:
        pickle.dump(attributions_all_batched, open(save, 'wb'))

    qe_model.net.roberta.encoder.output_attentions = False
    for l in qe_model.net.roberta.encoder.layer:
        l.attention.self.output_attentions = False
    return attributions_all_batched
