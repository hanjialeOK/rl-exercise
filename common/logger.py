import os
import tensorflow as tf
from collections import defaultdict


class Logger(object):

    def __init__(self, filename, summary_writer):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)  # count for compute mean
        self.summary_writer = summary_writer
        self.file = open(filename, 'w+t')
        self.keys = []

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self, timestep):
        newkeys = list(self.name2val.keys())
        if len(self.keys) == 0:
            self.keys.extend(newkeys)
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
        assert len(self.keys) == len(newkeys)
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = self.name2val.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=k, simple_value=float(v)) for k, v in self.name2val.items()]
        )
        self.summary_writer.add_summary(summary, timestep)
        self.summary_writer.flush()

        self.name2val.clear()

    def close(self):
        self.file.close()
        self.summary_writer.close()
