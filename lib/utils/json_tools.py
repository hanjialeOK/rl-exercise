import numpy as np
import random
import json


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def json_serializable(x):
    assert isinstance(x, dict)
    black_list = []
    for key in x.keys():
        if not is_jsonable(x[key]):
            black_list.append(key)
    for key in black_list:
        x.pop(key)
    return x


def set_global_seeds(i):
    myseed = int(i) % 1000
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)
