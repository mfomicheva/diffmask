import pickle
import torch

from tqdm.auto import trange

from diffmask.attributions.hidden_states_stats import hidden_states_statistics
from diffmask.utils.getter_setter import (
    roberta_getter,
    roberta_setter,
    bert_getter,
    bert_setter,
    gru_getter,
    gru_setter,
)

from diffmask.attributions.util import load_attributions


def schulz_loss(q_z_loc=None, q_z_scale=None, **kwargs):
    return {
        "q_z_loc": q_z_loc,
        "q_z_scale": q_z_scale,
        "loss_fn": lambda outputs, hidden_states, inputs_dict: outputs[0],
        "loss_kl_fn": lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
    }


def schulz_explainer(
    model,
    inputs_dict,
    getter,
    setter,
    q_z_loc,
    q_z_scale,
    loss_fn,
    loss_kl_fn,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
    verbose=False,
):

    with torch.no_grad():
        _, hidden_states = getter(model, inputs_dict)

    alpha = torch.full(
        hidden_states[hidden_state_idx].shape[:-1],
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)

    t = trange(steps) if verbose else range(steps)
    for _ in t:
        optimizer.zero_grad()
        gates = alpha.sigmoid()

        p_z_r = torch.distributions.Normal(
            loc=gates.unsqueeze(-1) * hidden_states[hidden_state_idx]
            + (1 - gates).unsqueeze(-1) * q_z_loc,
            scale=(q_z_scale + 1e-8) * (1 - gates).unsqueeze(-1),
        )

        q_z = torch.distributions.Normal(loc=q_z_loc, scale=(q_z_scale + 1e-8),)

        loss_model = loss_fn(
            *setter(
                model,
                inputs_dict,
                hidden_states=[None] * hidden_state_idx
                + [p_z_r.rsample()]
                + [None] * (len(hidden_states) - hidden_state_idx - 1),
            ),
            inputs_dict,
        )

        loss_kl = loss_kl_fn(
            torch.distributions.kl_divergence(p_z_r, q_z).mean(-1), inputs_dict
        )

        loss = loss_model + la * loss_kl

        loss.backward()
        optimizer.step()

        if verbose:
            t.set_postfix(
                loss="{:.2f}".format(loss.item()),
                loss_model="{:.2f}".format(loss_model.item()),
                loss_kl="{:.2f}".format(loss_kl.item()),
                refresh=False,
            )

    attributions = alpha.sigmoid().detach()

    return attributions


def qe_roberta_schulz_explainer(
        qe_model, tensor_dataset, text_dataset, verbose=False, save=None, load=None, input_only=True, steps=50,
        batch_size=1, num_layers=14, learning_rate=1e-1, aux_loss_weight=10, num_workers=20, hidden_states_stats=None,
):
    if load is not None:
        result = load_attributions(load)
        return result

    device = next(qe_model.parameters()).device
    if hidden_states_stats is not None:
        all_q_z_loc, all_q_z_scale = hidden_states_stats
    else:
        all_q_z_loc, all_q_z_scale = hidden_states_statistics(
            qe_model, qe_model.net.roberta, roberta_getter, input_only=input_only)
    result = []
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, num_workers=num_workers)
    for batch_idx, sample in enumerate(loader):
        input_ids, mask, _, labels = sample
        inputs_dict = {
            'input_ids': input_ids.to(device),
            'attention_mask': mask.to(device),
            'labels': labels.to(device),
        }
        all_attributions = []
        for layer_idx in range(num_layers):
            all_q_z_idx = 0 if input_only else layer_idx
            kwargs = schulz_loss(
                q_z_loc=all_q_z_loc[all_q_z_idx].unsqueeze(0).to(device),
                q_z_scale=all_q_z_scale[all_q_z_idx].unsqueeze(0).to(device),
                verbose=verbose,
            )
            layer_attributions = schulz_explainer(
                qe_model.net,
                inputs_dict=inputs_dict,
                getter=roberta_getter,
                setter=roberta_setter,
                hidden_state_idx=layer_idx,
                steps=steps,
                lr=learning_rate,
                la=aux_loss_weight,
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
    return result


def sst_bert_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    return schulz_explainer(
        model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=bert_getter,
        setter=bert_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )


def sst_gru_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    model.train()
    output = schulz_explainer(
        model.net,
        inputs_dict=inputs_dict,
        getter=gru_getter,
        setter=gru_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: model.training_step_end(
            outputs + (inputs_dict["labels"],)
        )["loss"],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["mask"]).mean(-1).mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
    model.eval()
    return output


def squad_bert_schulz_explainer(
    model,
    inputs_dict,
    all_q_z_loc,
    all_q_z_scale,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    device = next(model.parameters()).device
    return schulz_explainer(
        model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "start_positions": inputs_dict["start_positions"],
            "end_positions": inputs_dict["end_positions"],
        },
        getter=bert_getter,
        setter=bert_setter,
        q_z_loc=all_q_z_loc[hidden_state_idx].unsqueeze(0).to(device),
        q_z_scale=all_q_z_scale[hidden_state_idx].unsqueeze(0).to(device),
        loss_fn=lambda outputs, hidden_states, inputs_dict: outputs[0],
        loss_kl_fn=lambda kl, inputs_dict: (kl * inputs_dict["attention_mask"])
        .mean(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
