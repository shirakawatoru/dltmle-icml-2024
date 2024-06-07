from torch.utils.data import DataLoader, random_split

from .LAY.simple import SimpleSyntheticData
from .LAY.complex import ComplexSyntheticData
from .LAY.continuous import ContinuousSyntheticData

from .WAY.simplest import SimpleWAYSyntheticData

def load_data(rng, cfg):
    data_name = cfg['data']['name']
    params = cfg['data']['params']

    if data_name == 'lay-continuous':
        dataset = ContinuousSyntheticData(rng, params['n'], params['tau'])
    elif data_name == 'simple':
        dataset = SimpleSyntheticData(rng, params['n'], params['tau'])
    elif data_name == 'WAY':
        dataset = SimpleWAYSyntheticData(rng, params['n'])
    elif data_name == 'complex':
        dataset = ComplexSyntheticData(rng, params['n'], params['tau'], params['p'], params['lag'])
    else:
        raise ValueError(f'data_name not recognized: {data_name}')

    return dataset, LongitudinalDataLoader(dataset)

class LongitudinalDataLoader():
    def __init__(self, data, batch_size=64, ratio=[0.8, 0.2], **kwargs):
        self.batch_size = batch_size

        n = len(data)

        lengths = [int(n*r) for r in ratio]
        lengths[-1] = n - sum(lengths[:-1]) # adjust the last length to avoid rounding errors

        self.data = data
        self.train, self.val = random_split(data, lengths)

        self.train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def load_whole(self, shuffle=False):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=shuffle)

    def load_train(self, reshuffle=False):
        if reshuffle:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        return self.train_loader

    def load_train_wo_shuffle(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)

    def load_val(self):
        return self.val_loader
