# -*- coding: utf-8 -*-

import os

from .. import ingredient


@ingredient.capture
def lines(DATASETS_DIR, dataset, which_set, _log, filters=None, clean=None):
    _log.info('Loading lines [%s: %s]' % (dataset, which_set))
    lines = []
    for file in os.listdir(os.path.join(DATASETS_DIR, dataset, which_set)):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, file), 'r',
                  encoding='utf-8') as f:
            for line in f:
                if filters is not None:
                    if filters(line):
                        if clean is not None:
                            lines.append(clean(line.strip()))
                        else:
                            lines.append(line.strip())
                else:
                    if clean is not None:
                        lines.append(clean(line.strip()))
                    else:
                        lines.append(line.strip())
    return lines
