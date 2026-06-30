"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""


from rdkit import Chem
import os
import argparse
import pandas as pd
import json
from collections import Counter
from utils.dataset_config import RXN_KEY


def canonicalize_prod(p):
    import copy
    p = copy.deepcopy(p)
    p = canonicalize(p)
    p_mol = Chem.MolFromSmiles(p)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol)
    return p


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)


def fix_charge(mol):
    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    for atom in mol.GetAtoms():
        explicit_hs = atom.GetNumExplicitHs()
        charge = atom.GetFormalCharge()
        bond_vals = int(sum([b.GetBondTypeAsDouble()
                        for b in atom.GetBonds()]))
        if atom.GetSymbol() == 'O' and bond_vals == 1 and charge == -1 and explicit_hs == 0:
            if atom.GetNeighbors()[0].GetSymbol() != 'N':
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'N' and bond_vals == 1 and charge == 1 and explicit_hs == 3:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(2)

        if atom.GetSymbol() == 'N' and bond_vals == 0 and charge == 1 and explicit_hs == 4:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(3)

        if atom.GetSymbol() == 'N' and bond_vals == 2 and charge == 1 and explicit_hs == 2:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)

        if atom.GetSymbol() == 'S' and charge == -1 and explicit_hs == 0 and bond_vals == 1:
            atom.SetNumExplicitHs(1)
            atom.SetFormalCharge(0)
    return mol


def infer_correspondence(p):
    orig_mol = Chem.MolFromSmiles(p)
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))
    matches = list(canon_mol.GetSubstructMatches(orig_mol))
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()
                for atom in orig_mol.GetAtoms()}

    correspondence = {}
    if matches:
        for idx, match_idx in enumerate(matches[0]):
            match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()
            old_anum = idx_amap[idx]
            correspondence[old_anum] = match_anum
    return correspondence


def remap_rxn_smi(rxn_smi):
    r, p = rxn_smi.split(">>")
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))
    correspondence = infer_correspondence(p)

    rmol = Chem.MolFromSmiles(r)
    if rmol is None or rmol.GetNumAtoms() <= 1:
        return rxn_smi, None

    for atom in rmol.GetAtoms():
        atomnum = atom.GetAtomMapNum()
        if atomnum in correspondence:
            newatomnum = correspondence[atomnum]
            atom.SetAtomMapNum(newatomnum)

    max_amap = max([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])
    for atom in rmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    # fix simple atomic charge, eg. 'COO-', 'CH3O-', '(S=O)O-', '-NH3+', 'NH4+', 'NH2+', 'S-'
    rmol = fix_charge(rmol)
    canon_mol = fix_charge(canon_mol)

    rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol))
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(canon_mol)
    return rxn_smi_new, correspondence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k',
                        help='dataset: USPTO_50k, USPTO_full, or uspto_mit')
    parser.add_argument('--mode', type=str, default='train',
                        help='Type of dataset being prepared: train or valid or test')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    datadir = 'data/%s/' % args.dataset
    new_file = 'canonicalized_%s.csv' % args.mode
    filename = 'raw_%s.csv' % args.mode
    df = pd.read_csv(os.path.join(datadir, filename))
    print("Processing file of size: %d" % len(df))

    cols = ['id']
    if 'class' in df.columns:
        cols.append('class')
    cols.append(RXN_KEY)
    new_rows = []
    rejected = []
    reasons = Counter()
    for idx in range(len(df)):
        element = df.loc[idx]
        try:
            rxn_smi = element[RXN_KEY]
            rxn_smi_new, _ = remap_rxn_smi(rxn_smi)
            if rxn_smi_new is None:
                raise ValueError('canonicalization returned no reaction')
            row = {'id': element['id'], RXN_KEY: rxn_smi_new}
            if 'class' in df.columns:
                row['class'] = element['class']
            new_rows.append(row)
        except Exception as exc:
            reason = str(exc)
            reasons[reason] += 1
            row = element.to_dict()
            row['reason'] = reason
            rejected.append(row)
    pd.DataFrame(new_rows, columns=cols).to_csv(os.path.join(datadir, new_file), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(datadir, 'canonicalize_%s_rejected.csv' % args.mode), index=False)
    report = {'mode': args.mode, 'rows_read': int(len(df)), 'rows_kept': len(new_rows),
              'rows_rejected': len(rejected), 'rejection_counts_by_reason': dict(reasons)}
    json.dump(report, open(os.path.join(datadir, 'canonicalize_%s_report.json' % args.mode), 'w'), indent=2)


if __name__ == "__main__":
    main()
