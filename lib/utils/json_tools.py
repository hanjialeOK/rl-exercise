import numpy as np
import random
import json


def is_convert_json(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def convert_json(x):
    assert isinstance(x, dict)
    black_list = []
    for key in x.keys():
        if not is_convert_json(x[key]):
            black_list.append(key)
    for key in black_list:
        x.pop(key)
    return x
