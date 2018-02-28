# -*- coding: utf-8 -*-

import numpy as np

from ingredients.data import ingredients


@ingredients.config
def config():
    nb_words_normal = {
        'loc': 7,
        'scale': 1
    }
    word_length_normal = {
        'loc': 5,
        'scale': 1
    }


@ingredients.capture
def random_text(random_vocab, nb_words_normal, word_length_normal,
                max_len, batch_size):
    while True:
        sentences = []
        while len(sentences) < batch_size:
            nb_words = int(np.random.normal(**nb_words_normal))
            word_length = int(np.random.normal(**word_length_normal))
            s =  ' '.join([''.join(np.random.choice(random_vocab, word_length))
                for j in range(nb_words)
            ])

            if len(s) < max_len:
                sentences.append(s)
        yield sentences
