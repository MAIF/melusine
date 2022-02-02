import json
import os.path as op


def get_default_config(file):
    f = op.join(op.dirname(op.dirname(op.abspath(__file__))), "config", file)
    if op.exists(f):
        return json.load(open(f), encoding='utf-8')
    else:
        raise Exception(f"{file} not exist")
