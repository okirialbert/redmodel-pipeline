from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import json
import os
import tempfile
import base64
import numpy as np
import pandas as pd


host = 'localhost'
port = '8501'

_inference_timeout = 5


VOCAB_SIZE = 1000


data = ('If the output of the model contains only one named tensor, we omit the name and outputs key maps to a list of scalar or list values. If the model outputs multiple named tensors, we output an object instead. Each key of this object corresponds to a named output tensor. The format is similar to the request in column format mentioned above.')




def _infer_example(example):

    data = json.dumps( {"instances": [[example]]})
    
    server_url = 'http://' + host + ':' + port + '/v1/models/redmodel:predict'

    response = requests.post(
        server_url, data=data)
    response.raise_for_status()
    prediction = response.json()
    print(json.dumps(prediction, indent=4))
    return prediction


