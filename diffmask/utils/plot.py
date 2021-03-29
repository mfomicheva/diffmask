import math
import torch
import colorsys
from sty import bg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines


def _get_color(attr):
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 60
        sat = 100
        lig = 100 - int(70 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return [
        math.floor(e * 255)
        for e in colorsys.hls_to_rgb(hue / 360, lig / 100, sat / 100)
    ]


def print_attributions(tokens, attributions, labels=None, special=True, topk=None):

    if not special:
        attributions = torch.tensor(
            [e for e, w in zip(attributions, tokens) if w not in ("[CLS]", "[SEP]")]
        )
        tokens = [e for e in tokens if e not in ("[CLS]", "[SEP]")]

    def _print(text, values):
        print(
            " ".join(
                [
                    bg(*_get_color(a)) + w + bg.rs
                    for w, a in zip(text, values)
                ]
            )
        )
    if topk:
        attributions = torch.tensor(attributions)
        threshold, _ = torch.sort(attributions, descending=True)
        threshold = torch.full((len(attributions),), threshold[min(topk, len(attributions) - 1)])
        sparse_attributions = torch.zeros((len(attributions,)))
        sparse_attributions[attributions > threshold] = 1
        _print(tokens, sparse_attributions)
    else:
        _print(tokens, np.asarray(attributions) / max(attributions))
    if labels is not None:
        _print(tokens, labels)


def plot_sst_attributions(attributions, tokens, num_layers=14, name=None, save=False):
    fig = plt.figure(figsize=(9, len(tokens) / 3))
    fig.add_subplot(111, aspect=1.5)
    fig.patch.set_facecolor("white")
    plt.pcolormesh(
        (attributions / attributions.abs().max(0, keepdim=True).values).flip(0),
        edgecolors="k",
        linewidth=0.01,
    )
    plt.yticks(torch.arange(len(tokens)) + 0.5, reversed(tokens), size=16)
    plt.xticks(torch.arange(0, num_layers - 1, 3) + 0.5, ["E"] + list(range(3, num_layers - 1, 3)), size=16)

    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_squad_attributions(
    attributions,
    tokens,
    context,
    question,
    inputs_dict,
    logits_start_orig,
    logits_end_orig,
    name=None,
    save=False,
):

    fig = plt.figure(figsize=(9, len(tokens) / 3))
    ax = fig.add_subplot(111, aspect=2)
    fig.patch.set_facecolor("white")
    plt.pcolormesh(
        (attributions / attributions.abs().max(0, keepdim=True).values).flip(0),
        edgecolors="k",
        linewidth=0.01,
    )

    plt.yticks(
        torch.arange(len(tokens)) + 0.5,
        reversed(
            [
                r"$\bf{{{}}}$".format(e)
                if logits_start_orig.argmax(-1).item()
                <= i
                <= logits_end_orig.argmax(-1).item()
                else e
                for i, e in enumerate(tokens)
            ]
        ),
        size=16,
    )

    plt.xticks(torch.arange(0, 25, 3) + 0.5, ["E"] + list(range(3, 25, 3)), size=16)

    line = lines.Line2D([28, 28], [1, len(context) + 1], lw=4.0, color="black",)
    line.set_clip_on(False)
    ax.add_line(line)
    plt.text(
        27,
        len(context) / 2,
        " passage",
        rotation=90,
        fontsize=16,
        backgroundcolor="white",
    )

    line = lines.Line2D(
        [28, 28], [len(context) + 2, len(tokens) - 1], lw=4.0, color="black",
    )
    line.set_clip_on(False)
    ax.add_line(line)
    plt.text(
        27,
        len(context) + len(question) / 2,
        " question",
        rotation=90,
        fontsize=16,
        backgroundcolor="white",
    )

    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
