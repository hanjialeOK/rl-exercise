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

class History:
    def __init__(self):
        self.history = np.zeros(shape=(4, 84, 84), dtype=np.uint8)

    def add(self, screen):
        """
        screen: (84, 84)
        """
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        """
        return: (1, 4, 84, 84)
        """
        return np.expand_dims(self.history, axis=0)

    def reset(self):
        self.history *= 0