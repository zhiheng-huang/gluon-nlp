"""
Google Neural Machine Translation
=================================

This example shows how to implement the GNMT model with Gluon NLP Toolkit.

@article{wu2016google,
  title={Google's neural machine translation system:
   Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and
   Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and
   Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import io
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
import gluonnlp.data.batchify as btf
from gluonnlp.data import FixedBucketSampler, IWSLT2015
from gluonnlp.model import BeamSearchScorer
from encoder_decoder import get_gnmt_encoder_decoder
from translation import NMTModel, BeamSearchTranslator
from loss import SoftmaxCEMaskedLoss
from utils import logging_config
from bleu import compute_bleu
import utils

from s2sdata import S2SData
import _constants as _C
import pickle

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--train_dataset', type=str, default='train_si284', help='Train dataset to use.')
parser.add_argument('--val_dataset', type=str, default='dev_93', help='Validation dataset to use.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='vi', help='Target language')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--num_hidden', type=int, default=128, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--left_context', type=int, default=0, help='Left context size')
parser.add_argument('--right_context', type=int, default=0, help='Right context size')
parser.add_argument('--sub_sample', type=int, default=1, help='Sub sample rate')

parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=50, help='Maximum length of the source '
                                                                'sentence')
parser.add_argument('--tgt_max_len', type=int, default=50, help='Maximum length of the target '
                                                                'sentence')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpu', type=int, default=None,
                    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)
logging_config(args.save_dir)

num_workers = 1

train_data = S2SData(os.path.join("data/"+ args.train_dataset, "cmvn_by_len_2.scp"),
                     os.path.join("data/"+ args.train_dataset, "labels"),
                     None,
                     min_seq_length=0, max_seq_length=args.src_max_len,
                     max_label_length=args.tgt_max_len, filter_str=".",
                     left_context=args.left_context, right_context=args.right_context,
                     sub_sample=args.sub_sample, layout='NTC')
pickle.dump(train_data.dict(), open(os.path.join(args.save_dir, 'dict'), 'wb'))
val_data = S2SData(os.path.join("data/" + args.val_dataset, "cmvn_by_len_2.scp"),
                   os.path.join("data/" + args.val_dataset, "labels"),
                   os.path.join(args.save_dir, 'dict'),
                   min_seq_length=0, max_seq_length=args.src_max_len,
                   max_label_length=args.tgt_max_len, filter_str=".",
                   left_context=args.left_context, right_context=args.right_context,
                   sub_sample=args.sub_sample, layout='NTC')

train_loader = DataLoader(train_data, batch_size=args.batch_size, batchify_fn=train_data.batchify,
                          num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=args.batch_size, batchify_fn=val_data.batchify,
                        num_workers=num_workers)

if args.gpu is None:
    ctx = mx.cpu()
    print('Use CPU')
else:
    ctx = mx.gpu(args.gpu)

encoder, decoder = get_gnmt_encoder_decoder(hidden_size=args.num_hidden,
                                            dropout=args.dropout,
                                            num_layers=args.num_layers,
                                            num_bi_layers=args.num_bi_layers)
# model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
#                  embed_size=args.num_hidden, prefix='gnmt_')

model = NMTModel(src_vocab=None, tgt_vocab=train_data.dict(), encoder=encoder, decoder=decoder,
                 embed_size=args.num_hidden, prefix='gnmt_')
model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
model.hybridize()
logging.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=BeamSearchScorer(alpha=args.lp_alpha,
                                                          K=args.lp_k),
                                  max_length=args.tgt_max_len)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))


loss_function = SoftmaxCEMaskedLoss()
loss_function.hybridize()


def evaluate(data_loader):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    ground_truth = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0

    for _, (inst_ids, src_seq, src_valid_length, tgt_seq, tgt_valid_length) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        # Calculating Loss
        # import pdb; pdb.set_trace()
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        # all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        all_inst_ids.extend(inst_ids)
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(tgt_seq.shape[0]):
            # import pdb; pdb.set_trace()
            # ground_truth.append(tgt_seq[i][1:tgt_valid_length[i].astype(np.int32).asscalar() - 1].asnumpy())
            ground_truth.append(
                [train_data.dict().id_to_word(ele) for ele in
                 tgt_seq[i][1:(tgt_valid_length[i].astype(np.int32).asscalar() - 1)].asnumpy()])
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                # [tgt_vocab.idx_to_token[ele] for ele in
                [train_data.dict().id_to_word(ele) for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    # real_translation_out = [None for _ in range(len(all_inst_ids))]
    real_translation_out = {}
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    assert len(translation_out) == len(ground_truth)
    dis = 0.0
    total = 0.0
    for p, l in zip(translation_out, ground_truth):
        # print("prediction: %s" % p)
        # print("ground truth: %s" % l)
        dis += utils.edit_dis(p, l)
        total += len(l)

    logging.info('Count: {} dis: {} total: {} ACC: {:.4f}'
                 .format(len(translation_out), dis, total, 1 - float(dis) / total))
    return avg_loss, real_translation_out


def write_sentences(sentences, file_path):
    # with io.open(file_path, 'w', encoding='utf-8') as of:
    # with io.open(file_path, 'w') as of:
    with open(file_path, 'w') as of:
        for key in sentences:
            of.write(key + ' ' + ' '.join(sentences[key]) + '\n')
            # print(key, sentences[key])


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

    best_loss = float("Inf")
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time.time()
        # for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length)\
        #         in enumerate(train_data_loader):
        for batch_id, (_, src_seq, src_valid_length, tgt_seq, tgt_valid_length) \
                in enumerate(train_loader):
            # print("Training step %d" % step)
            # import pdb; pdb.set_trace()
            # print("[Train] (Epoch %d, Batch %d/%d) " % (epoch_id, batch_id + 1, len(train_loader)))
            # _, src_seq, src_valid_length, tgt_seq, tgt_valid_length = data  # labels_raw [[int]]

            # logging.info(src_seq.context) Context suddenly becomes GPU.
            src_seq = src_seq.as_in_context(ctx)
            tgt_seq = tgt_seq.as_in_context(ctx)
            src_valid_length = src_valid_length.as_in_context(ctx)
            tgt_valid_length = tgt_valid_length.as_in_context(ctx)
            with mx.autograd.record():
                out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
                loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
                loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
                loss.backward()
            grads = [p.grad(ctx) for p in model.collect_params().values()]
            gnorm = gluon.utils.clip_global_norm(grads, args.clip)
            trainer.step(1)
            src_wc = src_valid_length.sum().asscalar()
            tgt_wc = (tgt_valid_length - 1).sum().asscalar()
            step_loss = loss.asscalar()
            log_avg_loss += step_loss
            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc
            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     log_avg_gnorm / args.log_interval,
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0
        valid_loss, valid_translation_out = evaluate(val_loader)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}'
                     .format(epoch_id, valid_loss, np.exp(valid_loss)))
        write_sentences(valid_translation_out,
                        os.path.join(args.save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_path = os.path.join(args.save_dir, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_params(save_path)
        else:
            new_lr = trainer.learning_rate * args.lr_update_factor
            logging.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)
    model.load_params(os.path.join(args.save_dir, 'valid_best.params'))
    valid_loss, valid_translation_out = evaluate(val_loader)
    logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}'
                 .format(valid_loss, np.exp(valid_loss)))

    write_sentences(valid_translation_out,
                    os.path.join(args.save_dir, 'best_valid_out.txt'))


if __name__ == '__main__':
    train()
