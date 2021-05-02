from collections import defaultdict

from diffmask.utils.exceptions import TokenizationError


def map_bpe_moses(bpe_tokens, moses_tokens, sep='▁'):  # all special tokens need to be previously excluded from bpe_tokens
    bpe_tokens_orig = bpe_tokens
    bpe_tokens = []
    for bpe_tok in bpe_tokens_orig:
        if bpe_tok == sep:
            bpe_tokens.append(bpe_tok)
        else:
            bpe_tokens.append(bpe_tok.lstrip(sep))
    bpe_tokens = ' '.join(bpe_tokens)
    moses_tokens = ' '.join(moses_tokens)
    chrt_idx_bpe = 0
    chrt_idx_moses = 0
    tok_idx_bpe = 0
    tok_idx_moses = 0
    mapping = dict()
    while True:
        if chrt_idx_bpe >= len(bpe_tokens) or chrt_idx_moses >= len(moses_tokens):
            mapping[tok_idx_bpe] = tok_idx_moses
            break

        if bpe_tokens[chrt_idx_bpe] == moses_tokens[chrt_idx_moses] or bpe_tokens[chrt_idx_bpe] == '▁':
            if bpe_tokens[chrt_idx_bpe] == ' ':
                mapping[tok_idx_bpe] = tok_idx_moses
                tok_idx_bpe += 1
                tok_idx_moses += 1
            if bpe_tokens[chrt_idx_bpe] != '▁':
                chrt_idx_moses += 1
            chrt_idx_bpe += 1

        elif moses_tokens[chrt_idx_moses] == ' ':
            mapping[tok_idx_bpe] = tok_idx_moses
            tok_idx_moses += 1
            chrt_idx_moses += 1

        elif bpe_tokens[chrt_idx_bpe] == ' ':
            mapping[tok_idx_bpe] = tok_idx_moses
            tok_idx_bpe += 1
            chrt_idx_bpe += 1

        else:
            raise TokenizationError

    bpe_to_moses = mapping
    moses_to_bpe = defaultdict(list)
    for tok_idx_bpe, tok_idx_moses in mapping.items():
        moses_to_bpe[tok_idx_moses].append(tok_idx_bpe)
    return bpe_to_moses, moses_to_bpe


def map_bpe_moses_bert(bpe_tokens, moses_tokens, sep='##'):
    tok_idx_bpe = 0
    tok_idx_moses = 0
    mapping = dict()
    for i in range(len(bpe_tokens)):
        if i == len(bpe_tokens) - 1:
            mapping[tok_idx_bpe] = tok_idx_moses
            break
        mapping[tok_idx_bpe] = tok_idx_moses
        split_idx = i + 1 if sep == '##' else i
        if sep not in bpe_tokens[split_idx]:
            tok_idx_moses += 1
        tok_idx_bpe += 1
    bpe_to_moses = mapping
    moses_to_bpe = defaultdict(list)
    for tok_idx_bpe, tok_idx_moses in mapping.items():
        moses_to_bpe[tok_idx_moses].append(tok_idx_bpe)
    return bpe_to_moses, moses_to_bpe
