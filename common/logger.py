import os
import tensorflow as tf
from collections import defaultdict


class Logger(object):

    def __init__(self, filename1, filename2, summary_writer):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)  # count for logkv_mean
        self.key2str = {} # print info
        self.summary_writer = summary_writer
        self.file1 = open(filename1, 'w+t')
        self.file2 = open(filename2, 'w+t')
        self.keys = []

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = (oldval * cnt + val) / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def loginfo(self, key, val):
        if hasattr(val, '__float__'):
            valstr = '%-8.4g' % val
        else:
            valstr = str(val)
        self.key2str[self._truncate(key)] = self._truncate(valstr)

    def dumpkvs(self, timestep):
        # Write data to file1
        newkeys = list(self.name2val.keys() - self.keys)
        if newkeys:
            self.keys.extend(newkeys)
            self.file1.seek(0)
            lines = self.file1.readlines()
            self.file1.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file1.write(',')
                self.file1.write(k)
            self.file1.write('\n')
            for line in lines[1:]:
                self.file1.write(line[:-1])
                self.file1.write(',' * len(newkeys))
                self.file1.write('\n')
        # assert len(self.keys) == len(newkeys)
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file1.write(',')
            v = self.name2val.get(k)
            if v is not None:
                self.file1.write(str(v))
        self.file1.write('\n')
        self.file1.flush()

        # Tensorboard
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=k, simple_value=float(v)) for k, v in self.name2val.items()]
        )
        self.summary_writer.add_summary(summary, timestep)
        self.summary_writer.flush()

        # Print info and write info to file2
        for (key, val) in self.name2val.items():
            self.loginfo(key, val)
        # Find max widths
        assert len(self.key2str) > 0
        keywidth = max(map(len, self.key2str.keys()))
        valwidth = max(map(len, self.key2str.values()))
        # Write out the data
        dashes = '+' + '-' * (keywidth + valwidth + 5) + '+'
        lines = [dashes]
        for (key, val) in self.key2str.items():
            lines.append('| %s%s | %s%s |' % (
                key, ' ' * (keywidth - len(key)),
                val, ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        loginfo = '\n'.join(lines) + '\n'
        print(loginfo)
        self.file2.write(loginfo)
        self.file2.flush()

        self.name2val.clear()
        self.name2cnt.clear()
        self.key2str.clear()

    def _truncate(self, s):
        maxlen = 30
        return s[:maxlen-3] + '...' if len(s) > maxlen else s

    def close(self):
        self.file1.close()
        self.file2.close()
        self.summary_writer.close()
