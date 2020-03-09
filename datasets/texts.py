# -*- coding: utf-8 -*-

import math
import numpy as np
import os

from logging import Logger
from sacred import Ingredient
from tensorflow.keras.utils import Sequence
from typing import Dict, List, Optional, Tuple

from . import csv, json
from .. import paths


ingredient = Ingredient('datasets.texts', ingredients=[paths.ingredient])


class TextSequence(Sequence):
    def __init__(self, headline: List[List[int]], lead: List[List[int]],
                 text: List[List[int]], y_headline: List[List[int]],
                 y_lead: List[List[int]], y_text: List[List[int]],
                 headline_len: Optional[int] = None,
                 lead_len: Optional[int] = None,
                 text_len: Optional[int] = None,
                 total_len: Optional[int] = None, batch_size: int = 10,
                 mode: str = 'separate', dtype: type = np.uint):
        assert len(headline) == len(lead) and len(lead) == len(text) and \
            len(text) == len(y_headline) and len(y_headline) == len(y_lead) \
            and len(y_lead) == len(y_text)
        assert mode in ['separate', 'concat']

        self.headline = headline
        self.lead = lead
        self.text = text
        self.y_headline = y_headline
        self.y_lead = y_lead
        self.y_text = y_text
        self.headline_len = headline_len if headline_len else \
            max([len(i) for i in self.headline])
        self.lead_len = lead_len if lead_len else \
            max([len(i) for i in self.lead])
        self.text_len = text_len if text_len else \
            max([len(i) for i in self.text])
        self.total_len = total_len if total_len else \
            max([len(headline[i]) + len(lead[i]) + len(text[i])
                 for i in range(len(text))])
        self.batch_size = batch_size
        self.mode = mode
        self.dtype = dtype

    def __len__(self) -> int:
        return math.ceil(len(self.headline) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray],
                                             Dict[str, np.ndarray]]:
        if len(self.headline) >= (idx * self.batch_size) + self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = len(self.headline) - (idx * self.batch_size)
        start_idx = idx * self.batch_size
        end_idx = start_idx + current_batch_size

        if self.mode == 'separate':
            headline = np.zeros((current_batch_size, self.headline_len),
                                dtype=self.dtype)
            lead = np.zeros((current_batch_size, self.lead_len),
                            dtype=self.dtype)
            text = np.zeros((current_batch_size, self.text_len),
                            dtype=self.dtype)
            y_headline = np.zeros((current_batch_size, self.headline_len),
                                  dtype=self.dtype)
            y_lead = np.zeros((current_batch_size, self.lead_len),
                              dtype=self.dtype)
            y_text = np.zeros((current_batch_size, self.text_len),
                              dtype=self.dtype)
            for i, j in enumerate(range(start_idx, end_idx)):
                headline[i, 0:len(self.headline[j])] = self.headline[j]
                lead[i, 0:len(self.lead[j])] = self.lead[j]
                text[i, 0:len(self.text[j])] = self.text[j]
                y_headline[i, 0:len(self.y_headline[j])] = self.y_headline[j]
                y_lead[i, 0:len(self.y_lead[j])] = self.y_lead[j]
                y_text[i, 0:len(self.y_text[j])] = self.y_text[j]

            return {
                'headline': headline,
                'lead': lead,
                'text': text,
            }, {
                'pheadline': y_headline,
                'plead': y_lead,
                'ptext': y_text
            }
        elif self.mode == 'concat':
            text = np.zeros((current_batch_size, self.total_len),
                            dtype=self.dtype)
            y_text = np.zeros((current_batch_size, self.total_len),
                              dtype=self.dtype)
            for i, j in enumerate(range(start_idx, end_idx)):
                length = len(self.headline[j]) + len(self.lead[j]) + \
                    len(self.text[j])
                y_length = len(self.y_headline[j]) + len(self.y_lead[j]) + \
                    len(self.y_text[j])
                text[i, 0:length] = self.headline[j] + self.lead[j] + \
                    self.text[j]
                y_text[i, 0:y_length] = self.y_headline[j] + self.y_lead[j] + \
                    self.y_text[j]
            return {'text': text}, {'ptext': y_text}
        else:
            return {}, {}


@ingredient.capture
def get(dataset: str, batch_size: int, mode: str, headline_fieldname: str,
        lead_fieldname: str, text_fieldname: str, y_fieldname_prefix: str,
        append_one: bool, dtype: type, _log: Logger,
        vocab_path: Optional[str] = None) -> Tuple[TextSequence, TextSequence]:
    assert mode in ['separate', 'concat']

    fieldnames = [
        ('headline', headline_fieldname,
         f'{y_fieldname_prefix}{headline_fieldname}'),
        ('lead', lead_fieldname, f'{y_fieldname_prefix}{lead_fieldname}'),
        ('text', text_fieldname, f'{y_fieldname_prefix}{text_fieldname}')
    ]

    vocab = None
    if vocab_path:
        vocab = json.vocab(os.path.join('{datasets_dir}', 'heise', dataset,
                                        vocab_path))

    train_csv = os.path.join('{datasets_dir}', 'heise', dataset, 'train.csv')
    train_x, train_y = csv.load(train_csv, fieldnames, vocab=vocab,
                                append_one=append_one, dtype=dtype)

    val_csv = os.path.join('{datasets_dir}', 'heise', dataset, 'val.csv')
    val_x, val_y = csv.load(val_csv, fieldnames, vocab=vocab,
                            append_one=append_one, dtype=dtype)

    headline_len = max(max([len(i) for i in train_x['headline']]),
                       max([len(i) for i in val_x['headline']]))
    lead_len = max(max([len(i) for i in train_x['lead']]),
                   max([len(i) for i in val_x['lead']]))
    text_len = max(max([len(i) for i in train_x['text']]),
                   max([len(i) for i in val_x['text']]))
    total_len = max(max([len(train_x['headline'][i]) + len(train_x['lead'][i])
                         + len(train_x['text'][i]) for i in
                         range(len(train_x['text']))]),
                    max([len(val_x['headline'][i]) + len(val_x['lead'][i]) +
                         len(val_x['text'][i]) for i in
                         range(len(val_x['text']))]))

    return TextSequence(train_x['headline'], train_x['lead'], train_x['text'],
                        train_y['headline'], train_y['lead'], train_y['text'],
                        headline_len, lead_len, text_len, total_len,
                        batch_size, mode, dtype), \
        TextSequence(val_x['headline'], val_x['lead'], val_x['text'],
                     val_y['headline'], val_y['lead'], val_y['text'],
                     headline_len, lead_len, text_len, total_len, batch_size,
                     mode, dtype)
