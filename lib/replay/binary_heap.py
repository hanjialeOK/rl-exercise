from functools import total_ordering
import numpy as np
import random
import heapq

class ArrayBasedHeap():
    def __init__(self, size):
        self.array = []
        self.max_size = size
        self.sorted = []

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        assert idx < len(self.array)
        return self.array[idx]

    def __setitem__(self, idx, val):
        assert idx < len(self.array)
        self.array[idx] = val

    def __repr__(self):
        return str(self.array)

    # Only insert, No sort
    def insert(self, item):
        if len(self.array) < self.max_size:
            self.array.append(item)

    def sort(self):
        heap = heapq.heapify(self.array)
        self.sorted = heapq.nsmallest(len(self), heap)

    def sample(self):
        query_value = random.randint(0, len(self) - 1)
        item = self.sorted[query_value]
        return item.index, item.value

    def stratified_sample(self, batch_size):
        assert len(self.sorted) > 0
        bounds = np.linspace(0., 1., batch_size + 1)
        assert len(bounds) == batch_size + 1
        bounds = [int(x*len(self)) for x in bounds]
        segments = [(bounds[i], bounds[i+1]) for i in range(self.batch_size)]
        query_values = [random.randint(x[0], x[1]) for x in segments]
        items = [self.sorted[i] for i in query_values]
        indexs = [item.index for item in items]
        values = [item.value for item in items]
        return indexs, values

    def set(self, index, priority):
        pass


# for inserting into heap.
# index is into the replay buffer and refers to a specific state transition
@total_ordering     # generates missing compare methods
class HeapItem():
    def __init__(self, value, index):
        self.value = value
        self.index = index

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __repr__(self):
        return "(value={0}, index={1})".format(self.value, self.index)
