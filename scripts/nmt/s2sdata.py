# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import math
import re
from random import random

import mxnet.ndarray as nd
import numpy as np
from mxnet import context
from mxnet.gluon.data.dataset import Dataset

import kaldi_io
import pickle

BOS = '<bos>'
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'


class Dictionary(object):
    def __init__(self):
        self._word_to_idx = {}
        self._idx_to_word = []
        self.add_word(PAD)
        self.add_word(UNK)
        self.add_word(BOS)
        self.add_word(EOS)

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self._word_to_idx:
            self._idx_to_word.append(word)
            self._word_to_idx[word] = len(self._idx_to_word) - 1
        return self._word_to_idx[word]

    def __len__(self):
        return len(self._idx_to_word)

    def word_to_id(self, word):
        if word in self._word_to_idx:
            return self._word_to_idx[word]
        else:
            return self._word_to_idx[UNK]

    def words_to_ids(self, words):
        return [self.word_to_id(w) for w in words]

    def ids_to_words(self, ids):
        return [self.id_to_word(id) for id in ids]

    def id_to_word(self, id):
        assert id >= 0 and id < len(self._idx_to_word)
        return self._idx_to_word[id]


class S2SData(Dataset):
    def __init__(self, scp_file, label_file=None, dict_file=None, min_seq_length=None, max_seq_length=None,
                 max_label_length=None,
                 filter_str='.', left_context=0, right_context=0, sub_sample=1, context_pad=True, random_pad=False,
                 layout='TNC', reverse='False'):
        """
        Constructs a Dataset object for kaldi generated data. default output layout is 'TNC', 'TN' for labels.
        The layout of the scp file should be key, data:(seq length).

        :param scp_file: the index file for the dataset. two formats are supported: (key, path) and (key, path, length)
        :param label_file: label of the datasets, if supplied, will return a tuple of (features, labels).
        :param min_seq_length: minimum length in time dimension for input features. If not set,
                use minimum time in a batch.
        :param max_seq_length: maximum length in time dimension for input features. If not set,
                use maximum time in a batch.
        :param max_label_length: maximum length in time dimension for labels. If not set, use maximum time in a batch.
        :param filter_str: only iterate examples that match filter_str.
        :param left_context: number of previous time step's feature(s) to pack into channel dimension.
        :param right_context: number of future time step's feature(s) to pack into channel dimension.
        :param sub_sample: number of time step to skip. Must be greater than 0.
        :param context_pad: whether to 'edge' pad the sequence with right_context and left_context.
        :param random_pad: whether to pad the sequence with a random split between head and tail to match time
                dimension for a batch.
        :param layout: output layout ('TNC' or 'NTC')
        """
        if sub_sample < 1:
            raise Exception("sub_sample must be greater than 0")
        self._index = []
        self._labels = {}
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self._max_label_length = max_label_length
        self._left_context = left_context
        self._right_context = right_context
        self._sub_sample = sub_sample
        self._context_pad = context_pad
        self._random_pad = random_pad
        self._layout = layout
        self._reverse = reverse

        output_tokens = []
        if label_file and label_file is not '':
            with open(label_file, 'r') as f:
                for line in f:
                    key, label = line.strip().split(' ', 1)
                    label = label.split()  # list of str
                    if not max_label_length or len(label) <= self._max_label_length:
                        self._labels[key] = label
                        output_tokens.extend(label)

        if dict_file:
            self._dict = pickle.load(open(dict_file, 'rb'))
        else:
            self._dict = Dictionary()
            self._dict.add_words(output_tokens)

        count_total = 0
        count_used = 0
        with open(scp_file, 'r') as f:
            for line in f:
                count_total += 1
                if re.search(filter_str, line, re.I):
                    comps = line.strip().split()
                    if len(comps) < 3:
                        seq_len = kaldi_io.read_mat(comps[1]).shape[0]
                    else:
                        seq_len = int(comps[2])

                    if (min_seq_length and int(comps[2]) < min_seq_length) or (
                                max_seq_length and int(comps[2]) > max_seq_length):
                        continue
                    if len(self._labels) > 0 and (comps[0] not in self._labels):
                        continue
                    self._index.append((comps[0], comps[1], seq_len))
                    count_used += 1
        print("Total files: %d Used: %d\n" % (count_total, count_used))

    def __getitem__(self, idx):
        key, path, _ = self._index[idx]
        feat = kaldi_io.read_mat(path)
        shape = feat.shape
        window_size = 1 + self._left_context + self._right_context
        out = np.zeros((int(math.ceil(float(
            shape[0] - (
                (self._left_context + self._right_context) if not self._context_pad else 0)) / self._sub_sample)),
                        window_size * shape[1]))
        if self._left_context > 0 or self._right_context > 0:
            feat = np.pad(feat,
                          ((self._right_context if self._context_pad else 0,
                            self._right_context if self._context_pad else (shape[0] - window_size) % self._sub_sample),
                           (0, 0)), 'edge')
        out[:, self._left_context * shape[1]:(self._left_context + 1) * shape[1]] = \
            feat[self._left_context:feat.shape[0] - self._right_context:self._sub_sample]
        for i in range(self._left_context):
            # left context
            out[:, shape[1] * i:shape[1] * (i + 1)] = \
                feat[i:feat.shape[0] - self._left_context - self._right_context + i:self._sub_sample, :]

        for i in range(self._right_context):
            # right context
            out[:, shape[1] * (i + self._left_context + 1):shape[1] * (i + self._left_context + 2)] = \
                feat[self._left_context + 1 + i:feat.shape[0] - self._right_context + 1 + i:self._sub_sample, :]
        if len(self._labels) > 0:
            return key, out, self._labels[key]
        else:
            return key, out

    def __len__(self):
        return len(self._index)

    def dict(self):
        return self._dict

    def batchify(self, data):
        """
        Collate data into batch. Use shared memory for stacking.

        :param data: a list of array, with layout of 'NTC'.
        :return either x  and x's unpadded lengths, or x, x's unpadded lengths, y and y's unpadded lengths
                if labels are not supplied.
        """

        # input layout is NTC
        if len(self._labels) < 1:
            keys, inputs, labels = [item[0] for item in data], [item[1] for item in data], None
        else:
            keys, inputs, labels = [item[0] for item in data], [item[1] for item in data], \
                                   [item[2] for item in data]

        if len(data) > 1:
            max_data_len = max([seq.shape[0] for seq in inputs])
            max_labels_len = 0 if not labels else max([len(seq) for seq in labels])
        else:
            max_data_len = inputs[0].shape[0]
            max_labels_len = 0 if not labels else len(labels[0])

        x_lens = [item.shape[0] for item in inputs]
        if len(self._labels) < 1:
            y_lens = None
        else:
            y_lens = [len(item) + 2 for item in labels]  # 2 for BOS, EOS

        # labels = [None for i in range(len(inputs))]
        for i, seq in enumerate(inputs):
            if self._reverse.lower() == 'true':
                seq = np.flip(seq, axis=0)
            pad_len = max_data_len - seq.shape[0]
            if self._random_pad:
                head_pad = int(random() * pad_len)
                inputs[i] = np.pad(seq, ((head_pad, pad_len - head_pad), (0, 0)), 'constant', constant_values=0)
            else:
                inputs[i] = np.pad(seq, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
            if labels is not None:
                labels[i].insert(0, BOS)
                labels[i].append(EOS)
                while len(labels[i]) < max_labels_len + 2:  # 2 for BOS, EOS
                    labels[i].append(PAD)
                labels[i] = np.array(self._dict.words_to_ids(labels[i]))

        inputs = np.asarray(inputs, dtype=np.float32)
        if labels is not None:
            labels = np.asarray(labels, dtype=np.int32)
        if self._layout == 'TNC':
            inputs = inputs.transpose((1, 0, 2))
            if labels is not None:
                labels = labels.transpose((1, 0))
        elif self._layout == 'NTC':
            pass
        else:
            raise Exception("unknown layout")

        return (keys, nd.array(inputs, dtype=inputs.dtype, ctx=context.Context('cpu_shared', 0)),
                nd.array(x_lens, ctx=context.Context('cpu_shared', 0))) \
            if labels is None else (keys,
                                    nd.array(inputs, dtype=inputs.dtype, ctx=context.Context('cpu_shared', 0)),
                                    nd.array(x_lens, ctx=context.Context('cpu_shared', 0)),
                                    nd.array(labels, dtype=labels.dtype, ctx=context.Context('cpu_shared', 0)),
                                    nd.array(y_lens, ctx=context.Context('cpu_shared', 0)))
