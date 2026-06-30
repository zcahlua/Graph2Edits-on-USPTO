import argparse
import os

RXN_KEY = "reactants>reagents>production"
SPLITS = ("train", "valid", "test")


def normalize_dataset(dataset):
    return str(dataset).lower()


def is_large_dataset(dataset):
    return normalize_dataset(dataset) in ("uspto_full", "uspto_mit")


def supports_rxn_class(dataset):
    return normalize_dataset(dataset) == "uspto_50k"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def add_bool_arg(parser, name, default=False, help=None):
    parser.add_argument(name, nargs='?', const=True, default=default, type=str2bool, help=help)


def validate_rxn_class(dataset, use_rxn_class):
    if use_rxn_class and not supports_rxn_class(dataset):
        raise ValueError("--use_rxn_class is only supported for uspto_50k; %s has no default reaction-class labels" % dataset)


def input_csv_path(dataset, mode, input_stage='auto'):
    datadir = os.path.join('data', normalize_dataset(dataset))
    if input_stage == 'canonicalized':
        path = os.path.join(datadir, 'canonicalized_%s.csv' % mode)
    elif input_stage == 'raw':
        path = os.path.join(datadir, 'raw_%s.csv' % mode)
    elif input_stage == 'auto':
        canon = os.path.join(datadir, 'canonicalized_%s.csv' % mode)
        raw = os.path.join(datadir, 'raw_%s.csv' % mode)
        path = canon if os.path.exists(canon) else raw
    else:
        raise ValueError('Unknown input_stage: %s' % input_stage)
    return path
