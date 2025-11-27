import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "T": 2,
    "C": 3,
    "G": 4,
}

CHARPROTLEN = 4
# DNA:A/T/C/G  RNA:A/U/C/G

def graph_collate_func(x):
    d, p, y = zip(*x)
    # d = dgl.batch(d,dtype=torch.float)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p),dtype=torch.float), torch.tensor(y,dtype=torch.float)


def integer_label_DNA(sequence, max_length=110):   #DNA:110  RNA:450
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
