from random import shuffle

class DatasetBatcher():
    '''
    Returns batch of training dataset indices of size forward_batch_size.
    Shuffles the dataset after each whole pass.

    Equivalent to sampling using a dataloader with shuffle=True
    '''

    def __init__(self, dataset_size, forward_batch_size):
        self.dataset_size = dataset_size
        self.forward_batch_size = forward_batch_size
        # JFT TODO: Use less memory? Faster shuffle?
        self.dataset_idxs = range(self.dataset_size)
        shuffle(self.dataset_idxs)
        self.current_pos = 0

    def next(self):
        if self.current_pos + self.forward_batch_size < self.dataset_size:
            next_batch = self.dataset_idxs[self.current_pos:self.current_pos+self.forward_batch_size]
            self.current_pos += self.forward_batch_size
        else:
            # Special case: Epoch completed. Shuffle data before sampling
            next_batch = self.dataset_idxs[self.current_pos:]
            shuffle(self.dataset_idxs)
            remaining_batch_size = forward_batch_size - len(next_batch)
            next_batch += self.dataset_idxs[:remaining_batch_size]
            self.current_pos = remaining_batch_size

        assert(len(next_batch) == self.forward_batch_size)

        return next_batch



