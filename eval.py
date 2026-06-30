import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from collections import Counter
import torch
from rdkit import Chem, RDLogger

from models import Graph2Edits, BeamSearch
from utils.dataset_config import add_bool_arg, validate_rxn_class
lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def canonicalize_p(smi):
    p = canonicalize(smi)
    p_mol = Chem.MolFromSmiles(p)
    [a.SetAtomMapNum(a.GetIdx()+1) for a in p_mol.GetAtoms()]
    p_smi = Chem.MolToSmiles(p_mol)
    return p_smi


def canonical_reactant_set(smi):
    try:
        frags = []
        for frag in smi.split('.'):
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol is None:
                return None
            frags.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        return set(frags)
    except Exception:
        return None


def _highest_epoch_checkpoint(exp_dir):
    epochs = [f for f in os.listdir(exp_dir) if f.startswith('epoch_') and f.endswith('.pt')]
    epochs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not epochs:
        raise IOError('No checkpoints found in %s; expected best.pt, latest.pt, or epoch_*.pt' % exp_dir)
    return os.path.join(exp_dir, epochs[-1])


def resolve_checkpoint(exp_dir, spec):
    if spec == 'best':
        best = os.path.join(exp_dir, 'best.pt')
        if os.path.exists(best):
            return best
        latest = os.path.join(exp_dir, 'latest.pt')
        if os.path.exists(latest):
            return latest
        return _highest_epoch_checkpoint(exp_dir)
    if spec == 'latest':
        latest = os.path.join(exp_dir, 'latest.pt')
        if os.path.exists(latest):
            return latest
        return _highest_epoch_checkpoint(exp_dir)
    if spec.startswith('epoch_') and spec.endswith('.pt'):
        return os.path.join(exp_dir, spec)
    return spec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k',
                        help='dataset: USPTO_50k or USPTO_full')
    add_bool_arg(parser, '--use_rxn_class', default=False, help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='27-06-2022--10-27-22',
                        help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    parser.add_argument('--checkpoint', default='best', help='best, latest, epoch_N.pt, or path')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    try:
        validate_rxn_class(args.dataset, args.use_rxn_class)
    except ValueError as exc:
        parser.error(str(exc))

    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)
    if args.use_rxn_class:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
    else:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')

    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['saveables']

    model = Graph2Edits(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    pred_file = os.path.join(exp_dir, 'pred_results.txt')
    file_num = 1
    while os.path.exists(pred_file):
        pred_file = os.path.join(exp_dir, f'pred_results_{file_num}.txt')
        file_num += 1

    with open(pred_file, 'a') as fp:
        for idx in p_bar:
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class
            edit_steps = len(rxn_data.edits)
            counter.append(edit_steps)

            r, p = rxn_smi.split('>>')
            r_set = canonical_reactant_set(r)

            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

            fp.write(f'({idx}) {rxn_smi}\n')

            beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = canonical_reactant_set(pred_smi)
                correct = pred_set is not None and pred_set == r_set
                str_edits = '|'.join(f'({str(edit)};{p})'for edit, p in zip(
                    path['rxn_actions'], path['edits_prob']))
                fp.write(
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')
                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True

            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)

            for edit in rxn_data.edits:
                if edit != 'Terminate' and len(edit) > 1 and (edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (2, 2) or edit[1] == (2, 3)):
                    stereo_rxn.append(idx)
                    if beam_matched:
                        stereo_rxn_cor.append(idx)

            msg = 'average score'
            for beam_idx in [k for k in [1, 3, 5, 10] if k <= args.beam_size]:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.4f' % (beam_idx, match_acc)
            p_bar.set_description(msg)

        edit_steps = Counter(counter)
        edit_steps_correct = Counter(edit_steps_cor)
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')
        fp.write(
            f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))
        summary = 'final top-k: ' + ', '.join(['t%d %.4f' % (k, np.sum(top_k[:k]) / max(1, len(test_data))) for k in [x for x in [1,3,5,10] if x <= args.beam_size]])
        print(summary)
        fp.write(summary + '\n')


if __name__ == '__main__':
    main()
