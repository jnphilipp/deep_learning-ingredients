# -*- coding: utf-8 -*-

import os
import re

from .. import ingredient


@ingredient.capture
def lines(DATASETS_DIR, dataset, which_set, filters=None, clean=None,
          nb_lines=None, _log=None):
    _log.info('Loading lines [%s: %s]' % (dataset, which_set))

    lines = []
    vocab = set()
    max_len = 0
    for file in os.listdir(os.path.join(DATASETS_DIR, dataset, which_set)):
        with open(os.path.join(DATASETS_DIR, dataset, which_set, file), 'r',
                  encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if filters is not None:
                    if filters(line):
                        if clean is not None:
                            lines.append(clean(line))
                        else:
                            lines.append(line)
                    else:
                        line = None
                else:
                    if clean is not None:
                        lines.append(clean(line))
                    else:
                        lines.append(line)
                if line is not None:
                    vocab = vocab.union(set(lines[-1]))
                    max_len = max(len(lines[-1]), max_len)
                if nb_lines is not None and nb_lines <= len(lines):
                    break
        if nb_lines is not None and nb_lines <= len(lines):
            break
    return lines, vocab, max_len


@ingredient.capture
def generate(source, target, lines_per_file=250000, _log=None):
    lines = set()
    vocab = set(' .,:;!?()^/][{}’_+”#*´@$€%&‘`…<>§†©×°=®™"-')
    for file in os.listdir(source):
        with open(os.path.join(source, file), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('• '):
                    line = line.replace('• ', '')

                new_line = ''
                for s in line:
                    if s.isalnum():
                        new_line += s
                        vocab.add(s)
                    elif s in '"\'„“”«»‚‘″':
                        new_line += '"'
                    elif s in '-—−–':
                        new_line += '-'
                    elif s in '.,:;!?()^/][{}’_+”#*´@$€%&‘`…<>§†©×°=®™±':
                        new_line += s
                    elif s.isspace():
                        new_line += ' '
                    elif not s.isprintable():
                        continue
                    else:
                        new_line = None
                        removed += 1
                        break

                if new_line is not None:
                    line = re.sub(r'\s\s+', ' ', new_line)
                else:
                    continue
                lines.add(line)

    lines = sorted(list(lines), key=str.lower)
    for i in range(int(len(lines) / lines_per_file) + 1):
        with open(os.path.join(target, f'{i:03d}.txt'), 'w',
                  encoding='utf-8') as f:
            start = i * lines_per_file
            for j in range(start, min(start + lines_per_file, len(lines))):
                f.write(lines[j])
                f.write('\n')

    print('lines', lines)
    print(vocab, len(vocab))
    return lines
