import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from utils import integer_label_protein, integer_label_peptide


class PepPIDataset(data.Dataset):
    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df = df

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['Peptide']
        v_d = integer_label_peptide(v_d)

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        # y = torch.Tensor([y])
        return v_d, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
