import os
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


def save_json(x, base_dir):
    assert isinstance(x, dict)
    if not os.path.exists(base_dir):
        raise
    config_json = json.dumps(x, sort_keys=False,
                             indent=4, separators=(',', ': '))
    with open(os.path.join(base_dir, "config.json"), 'w') as out:
        out.write(config_json)
