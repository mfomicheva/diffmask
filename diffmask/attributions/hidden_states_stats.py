import torch

from tqdm import tqdm

from diffmask.utils.getter_setter import (
    gru_getter,
    toy_getter,
)


def hidden_states_statistics(model, pretrained_model, getter, input_only):
    return transformers_hidden_states_statistics(model, pretrained_model, getter, input_only=input_only)


def transformers_hidden_states_statistics(model, transformers_model, getter, input_only=True):

    with torch.no_grad():
        all_hidden_states = []
        for batch in tqdm(model.train_dataloader()):
            batch = tuple(e.to(next(model.parameters()).device) for e in batch)
            if input_only:
                hidden_states = [transformers_model.embeddings.word_embeddings(batch[0])]
            else:
                _, hidden_states = getter(
                    model.net,
                    {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        **({"token_type_ids": batch[2]} if len(batch) == 5 else {}),
                    },
                )
            all_hidden_states.append(torch.stack(hidden_states).cpu())

        all_q_z_loc = sum([e.sum(1) for e in all_hidden_states]) / sum(
            [e.shape[1] for e in all_hidden_states]
        )
        all_q_z_scale = (
            sum(((all_q_z_loc.unsqueeze(1) - e) ** 2).sum(1) for e in all_hidden_states)
            / sum([e.shape[1] for e in all_hidden_states])
        ).sqrt()

    return all_q_z_loc, all_q_z_scale


def sst_gru_hidden_states_statistics(model):

    with torch.no_grad():
        all_hidden_states = []
        for batch in tqdm(model.train_dataloader()):
            batch = tuple(e.to(next(model.parameters()).device) for e in batch)
            _, hidden_states = gru_getter(
                model.net, {"input_ids": batch[0], "mask": batch[1],}
            )
            all_hidden_states.append(torch.stack(hidden_states).cpu())

        all_hidden_states = torch.cat(all_hidden_states, 1)
        all_q_z_loc = all_hidden_states.mean(1)
        all_q_z_scale = all_hidden_states.std(1)

        return all_q_z_loc, all_q_z_scale


def toy_hidden_states_statistics(model):

    all_hidden_states = [[], [], []]
    for batch in tqdm(model.train_dataloader()):
        batch = tuple(e.to(next(model.parameters()).device) for e in batch)
        _, hidden_states = toy_getter(
            model, {"query_ids": batch[0], "input_ids": batch[1], "mask": batch[2],}
        )
        all_hidden_states[0].append(hidden_states[0].cpu())
        all_hidden_states[1].append(hidden_states[1].cpu())
        all_hidden_states[2].append(hidden_states[2].cpu())

    all_hidden_states = [torch.cat(e, 0) for e in all_hidden_states]
    all_q_z_loc = [e.mean(0) for e in all_hidden_states]
    all_q_z_scale = [e.std(0) for e in all_hidden_states]

    return all_q_z_loc, all_q_z_scale
