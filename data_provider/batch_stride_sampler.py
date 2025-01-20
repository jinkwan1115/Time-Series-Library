from torch.utils.data import Sampler
import random

class StrideSampler(Sampler):
    def __init__(self, args, data_source, stride):
        """
        Args:
            data_source (Dataset): The dataset to sample from.
            stride (int): The gap between consecutive indices in a batch.
            batch_size (int): Number of samples in a batch.
        """
        self.args = args
        self.data_source = data_source
        self.stride = stride
        self.batch_size = self.args.batch_size
        self.indices = [
            i for i in range(0, len(self.data_source) - self.stride * (self.batch_size - 1) + 1)
        ]

    def __iter__(self):
        shuffled_indices = self.indices.copy()
        random.shuffle(shuffled_indices)

        for start_idx in shuffled_indices:
            # batch_indices = [start_idx + i * self.stride for i in range(self.batch_size)]
            # print(batch_indices)
            # for idx in batch_indices:
            #     yield idx
            for i in range(self.batch_size):
                yield start_idx + i * self.stride

    def __len__(self):
        return len(self.indices)

