# -*- coding: utf-8 -*-

from logging import Logger
from tensorflow.keras import backend as B
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework.ops import Tensor
from typing import Dict, List, Optional, Union

from . import ingredient
from .. import models


@ingredient.capture
def build(inner_net_type: str, outputs: Dict, optimizer: Optimizer,
          _log: Logger, loss_weights: Optional[Union[List, Dict]] = None,
          sample_weight_mode: Optional[Union[str, Dict[str, str],
                                             List[str]]] = None,
          weighted_metrics: Optional[List] = None,
          target_tensors: Optional[Union[Tensor, List[Tensor]]] = None, *args,
          **kwargs) -> Model:
    if 'name' in kwargs:
        name = kwargs.pop('name')
    else:
        name = 'siamese'
    _log.info(f'Build Siamese [{inner_net_type}] model [{name}]')

    inner_model = models.get(None, inner_net_type,
                             outputs=[{'t': 'vec', 'loss': 'mse'}], *args,
                             **kwargs)

    input_r = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_r')
    input_l = Input(inner_model.get_layer('input').input_shape[1:],
                    name='input_l')

    xr = inner_model(input_r)
    xl = inner_model(input_l)

    # outputs
    output_types = ['distance']
    assert set([o['t'] for o in outputs]).issubset(output_types)

    outs = []
    loss = []
    metrics = {}
    for output in outputs:
        loss.append(output['loss'])
        if 'metrics' in output:
            metrics[output['name']] = output['metrics']

        if output['t'] == 'distance':
            outs.append(Lambda(lambda x: B.mean(B.abs(x[0] - x[1]), axis=-1),
                               name=output['name'],
                               output_shape=(1,))([xr, xl]))

    siamese_model = Model(inputs=[input_r, input_l], outputs=outs, name=name)
    siamese_model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                          loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode,
                          weighted_metrics=weighted_metrics,
                          target_tensors=target_tensors)
    return inner_model, siamese_model
