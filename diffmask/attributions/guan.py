import torch
import pickle

from tqdm.auto import trange

from diffmask.utils.getter_setter import (
    roberta_getter,
    roberta_setter,
    bert_getter,
    bert_setter,
    gru_getter,
    gru_setter,
)


def guan_loss(**kwargs):
    return {
        "s_fn": lambda outputs, hidden_states: outputs[1:],
        "loss_l2_fn": lambda s, inputs_dict: sum(s_i.sum(-1).mean(-1) for s_i in s),
        "loss_h_fn": lambda h, inputs_dict: (h * inputs_dict["attention_mask"])
        .sum(-1)
        .mean(-1),
    }


def guan_explainer(
    model,
    inputs_dict,
    getter,
    setter,
    s_fn,
    loss_l2_fn,
    loss_h_fn,
    hidden_state_idx=0,
    steps=10,
    lr=1e-1,
    la=10,
):
    with torch.no_grad():
        outputs, hidden_states = getter(model, inputs_dict)
        s = s_fn(outputs, hidden_states)

    sigma = torch.full(
        hidden_states[hidden_state_idx].shape[:-1],
        -5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    optimizer = torch.optim.RMSprop([sigma], lr=lr, centered=True)

    t = trange(steps)
    for _ in t:
        optimizer.zero_grad()

        eps = torch.distributions.Normal(
            loc=torch.zeros_like(sigma), scale=torch.nn.functional.softplus(sigma),
        )

        noise = eps.rsample((hidden_states[hidden_state_idx].shape[-1],)).permute(
            list(range(1, len(hidden_states[hidden_state_idx].shape))) + [0]
        )

        s_pred = s_fn(
            *setter(
                model,
                inputs_dict,
                hidden_states=[None] * hidden_state_idx
                + [hidden_states[hidden_state_idx] + noise]
                + [None] * (len(hidden_states) - hidden_state_idx - 1),
            )
        )

        loss_l2 = loss_l2_fn(
            [(s_i - s_pred_i) ** 2 for s_i, s_pred_i in zip(s, s_pred)], inputs_dict
        )
        loss_h = loss_h_fn(eps.entropy(), inputs_dict)

        loss = loss_l2 - la * loss_h

        loss.backward()
        optimizer.step()

        t.set_postfix(
            loss="{:.2f}".format(loss.item()),
            loss_l2="{:.2f}".format(loss_l2.item()),
            loss_h="{:.2f}".format(-loss_h.item()),
            refresh=False,
        )

    return torch.nn.functional.softplus(sigma).detach()


def qe_roberta_guan_explainer(
        qe_model, tensor_dataset, save=None, load=None, steps=50, batch_size=1, num_layers=14, learning_rate=1e-1,
        aux_loss_weight=10, verbose=False, num_workers=20
):

    if load is not None:
        result = pickle.load(open(load, 'rb'))
        return result

    device = next(qe_model.parameters()).device
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
            kwargs = guan_loss()
            layer_attributions = guan_explainer(
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


def sst_bert_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    return guan_explainer(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "labels": inputs_dict["labels"],
        },
        getter=bert_getter,
        setter=bert_setter,
        s_fn=lambda outputs, hidden_states: outputs[1:],
        loss_l2_fn=lambda s, inputs_dict: sum(s_i.sum(-1).mean(-1) for s_i in s),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["attention_mask"])
        .sum(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )


def sst_gru_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    device = next(model.parameters()).device
    model.train()
    output = guan_explainer(
        model=model.net,
        inputs_dict=inputs_dict,
        getter=gru_getter,
        setter=gru_setter,
        s_fn=lambda outputs, hidden_states: outputs,
        loss_l2_fn=lambda s, inputs_dict: sum(s_i.sum(-1).mean(-1) for s_i in s),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["mask"]).sum(-1).mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
    model.eval()
    return output


def squad_bert_guan_explainer(
    model, inputs_dict, hidden_state_idx=0, steps=10, lr=1e-1, la=10,
):
    return guan_explainer(
        model=model.net,
        inputs_dict={
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["mask"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "start_positions": inputs_dict["start_positions"],
            "end_positions": inputs_dict["end_positions"],
        },
        getter=bert_getter,
        setter=bert_setter,
        s_fn=lambda outputs, hidden_states: outputs[1:],
        loss_l2_fn=lambda s, inputs_dict: sum(
            (s_i * inputs_dict["attention_mask"]).sum(-1).mean(-1) for s_i in s
        ),
        loss_h_fn=lambda h, inputs_dict: (h * inputs_dict["attention_mask"])
        .sum(-1)
        .mean(-1),
        hidden_state_idx=hidden_state_idx,
        steps=steps,
        lr=lr,
        la=la,
    )
