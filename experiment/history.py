# -*- coding: utf-8 -*-

import json
import os

from ingredients.experiment import ingredients


@ingredients.capture
def save(path, name, history):
    with open(os.path.join(path, '%s.json' % name), 'w', encoding='utf8') as f:
        f.write(json.dumps(history, indent=4))
        f.write('\n')
