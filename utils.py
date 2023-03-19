import torch
class Accumulator:
    def __init__(self, n) -> None:
        self.data = [0.0] * n
    
    def __getitem__(self, idx):
        return self.data[idx]

    def add(self, *args):
        assert len(args) == len(self.data)
        self.data = [float(ori + new) for (ori, new) in zip(self.data, args)]
    
    def empty(self):
        self.data = [0.0] * len(self.data)