"""WikiText-2 as a sequence of characters."""
import numpy as np
import torch


def load_wikitext(path):
    """Read the character-level WikiText-2 file.

    ``wikitext2_chars.npz`` holds WikiText-2 (raw, v1; Merity et al., 2016;
    CC BY-SA 3.0) as character ids: characters seen at least 50 times in the
    training text keep their own id, the rest become ``'?'``.

    Returns ``{'vocab': list of str, 'train', 'validation', 'test': int64
    tensors of ids}``.
    """
    data = np.load(path)
    result = {'vocab': [str(char) for char in data['vocab']]}
    for split in ('train', 'validation', 'test'):
        result[split] = torch.from_numpy(data[split].astype(np.int64))
    return result


def encode(text, vocab):
    """Ids of the characters of ``text``; unknown characters become ``'?'``."""
    index = {char: position for position, char in enumerate(vocab)}
    return torch.tensor([index.get(char, index['?']) for char in text])


def decode(ids, vocab):
    return ''.join(vocab[int(i)] for i in ids)


class CharWindowDataset:
    """Random windows of a long text for next-character prediction.

    ``offsets`` (``n_windows`` random start positions, fixed by ``seed``)
    are given. Item ``index`` is the window starting at
    ``offsets[index]``: ``{'tokens': ids[o : o + context], 'targets':
    ids[o + 1 : o + context + 1]}``, the targets shifted by one character.
    """

    def __init__(self, ids, context=128, n_windows=1000, seed=0):
        self.ids = ids
        self.context = context
        generator = torch.Generator().manual_seed(seed)
        self.offsets = torch.randint(0, len(ids) - context - 1, (n_windows,), generator=generator)

    def __len__(self):
        res = 1
        ## YOUR CODE HERE
        return res

    def __getitem__(self, index):
        sample = {
            'tokens': torch.zeros(self.context, dtype=torch.long),
            'targets': torch.zeros(self.context, dtype=torch.long),
        }
        ## YOUR CODE HERE
        return sample
