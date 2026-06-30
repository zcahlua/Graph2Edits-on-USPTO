#!/usr/bin/env python
import argparse, csv, json, os, random, zipfile, tempfile, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict
from utils.dataset_config import RXN_KEY, SPLITS, str2bool

Chem = None
pd = None

def require_runtime_deps():
    global Chem, pd
    if Chem is None:
        from rdkit import Chem as _Chem
        Chem = _Chem
    if pd is None:
        import pandas as _pd
        pd = _pd

REACTION_COLUMNS = [RXN_KEY, 'reactions', 'reaction', 'rxn', 'rxn_smiles', 'reaction_smiles']

def split_rxn(s):
    s = str(s).strip()
    if '>>' in s and s.count('>>') == 1:
        r,p = s.split('>>'); return r, '', p, 'graph2edits'
    if s.count('>') >= 2:
        parts = s.split('>')
        return parts[0], parts[1], '>'.join(parts[2:]), 'three_part'
    raise ValueError('reaction cannot be parsed as reactants>>product or reactant>reagent>product')

def mol_from(s):
    return Chem.MolFromSmiles(s) if s else None

def validate_and_convert(row, include_reagents, allow_multi_product):
    raw = row['reaction']
    r, reag, p, kind = split_rxn(raw)
    if not r or not p: raise ValueError('reactants or product are missing')
    if '.' in p and not allow_multi_product: raise ValueError('multiple product fragments')
    pmol = mol_from(p)
    if pmol is None: raise ValueError('product molecule cannot be parsed')
    if pmol.GetNumAtoms() <= 1 or pmol.GetNumBonds() <= 1: raise ValueError('product too small for Graph2Edits preprocessing')
    rmol = mol_from(r if not (include_reagents and reag) else r + '.' + reag)
    if rmol is None: raise ValueError('reactant molecule cannot be parsed')
    p_maps=[]
    for a in pmol.GetAtoms():
        m = a.GetAtomMapNum()
        if not isinstance(m, int) or m == 0: raise ValueError('product atoms have missing/zero/non-integer atom-map numbers')
        p_maps.append(m)
    if len(p_maps) != len(set(p_maps)): raise ValueError('duplicate product atom-map numbers')
    r_maps = [a.GetAtomMapNum() for a in rmol.GetAtoms()]
    nz = [m for m in r_maps if m]
    if len(nz) != len(set(nz)): raise ValueError('reactant side has duplicate nonzero atom-map numbers')
    if not set(p_maps).issubset(set(nz)): raise ValueError('product atom-map numbers are not a subset of nonzero reactant atom-map numbers')
    # RDKit canonicalization smoke
    if Chem.MolToSmiles(pmol) is None or Chem.MolToSmiles(rmol) is None: raise ValueError('RDKit canonicalization fails')
    rr = r if not (include_reagents and reag) else r + '.' + reag
    return rr + '>>' + p, {'product_atoms': len(p_maps), 'reactant_zero_maps': sum(1 for m in r_maps if m == 0)}

def detect_split(path):
    n=os.path.basename(path).lower()
    if 'train' in n: return 'train'
    if 'valid' in n or 'val' in n or 'dev' in n: return 'valid'
    if 'test' in n: return 'test'
    return None

def extract_inputs(inp):
    if os.path.isdir(inp):
        return [os.path.join(dp,f) for dp,_,fs in os.walk(inp) for f in fs if not f.startswith('.')]
    if zipfile.is_zipfile(inp):
        td=tempfile.mkdtemp(prefix='uspto_mit_zip_'); zipfile.ZipFile(inp).extractall(td)
        return [os.path.join(dp,f) for dp,_,fs in os.walk(td) for f in fs], td
    return [inp]

def read_file(path, fmt, reaction_column=None):
    rows=[]; ext=os.path.splitext(path)[1].lower()
    if fmt in ('deepchem','graph2edits') or ext in ('.csv','.tsv'):
        sep='\t' if ext=='.tsv' else ','
        try: df=pd.read_csv(path, sep=sep)
        except Exception:
            df=None
        if df is not None and len(df.columns):
            col = reaction_column or next((c for c in REACTION_COLUMNS if c in df.columns), None)
            if col is None: col = next((c for c in df.columns if 'reaction' in c.lower() or 'rxn' in c.lower()), None)
            if col:
                for i,row in df.iterrows():
                    d={'reaction': row[col], 'id': row.get('id', None)}
                    if 'class' in df.columns: d['class']=row['class']
                    rows.append(d)
                return rows
    with open(path) as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith('#'): continue
            rows.append({'reaction': line.split()[0], 'id': None})
    return rows

def random_split_rows(rows, train_frac, valid_frac, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n >= 3:
        n_valid = max(1, int(round(n * valid_frac)))
        n_test = max(1, int(round(n * (1.0 - train_frac - valid_frac))))
        if n_valid + n_test >= n:
            n_valid = 1
            n_test = 1
        n_train = n - n_valid - n_test
    else:
        n_train = int(n * train_frac)
        n_valid = int(n * valid_frac)
        n_test = n - n_train - n_valid
    return {'train': rows[:n_train], 'valid': rows[n_train:n_train+n_valid], 'test': rows[n_train+n_valid:n_train+n_valid+n_test]}


def assert_nonempty_splits(result, allow_empty_splits):
    empty = [split for split in SPLITS if len(result[split]) == 0]
    if empty and not allow_empty_splits:
        raise SystemExit('Empty output split(s): %s. Provide train/valid/test inputs, more rows, or pass --allow_empty_splits true for smoke/debug use.' % ', '.join(empty))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--format', choices=['auto','jin','deepchem','graph2edits'], default='auto')
    p.add_argument('--out', default='data/uspto_mit'); p.add_argument('--include_reagents', type=str2bool, default=False)
    p.add_argument('--split', choices=['predefined','random'], default='predefined'); p.add_argument('--train_frac', type=float, default=0.8); p.add_argument('--valid_frac', type=float, default=0.1); p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--limit_rows', type=int); p.add_argument('--allow_multi_product', type=str2bool, default=False); p.add_argument('--allow_empty_splits', type=str2bool, default=False); p.add_argument('--reaction_column')
    a=p.parse_args(); require_runtime_deps(); os.makedirs(a.out, exist_ok=True)
    tmp=None; result={s:[] for s in SPLITS}; all_rows=[]; report={'files_read':[], 'rows_read':0, 'rows_kept':0, 'rows_rejected':0, 'rejection_counts_by_reason':{}, 'atom_map_statistics':{}, 'include_reagents':a.include_reagents}
    try:
        ex=extract_inputs(a.input); files,tmp=(ex if isinstance(ex, tuple) else (ex,None))
        predefined=False
        for f in files:
            sp=detect_split(f); rows=read_file(f, a.format, a.reaction_column); report['files_read'].append(f); report['rows_read']+=len(rows)
            for row in rows:
                if a.limit_rows is not None and len(all_rows) >= a.limit_rows:
                    break
                row['_split']=sp; all_rows.append(row)
                if sp: predefined=True
            if a.limit_rows is not None and len(all_rows) >= a.limit_rows:
                break
        if a.split=='predefined' and predefined:
            split_rows={s:[r for r in all_rows if r['_split']==s] for s in SPLITS}; report['split_method']='predefined'
        else:
            split_rows=random_split_rows(all_rows, a.train_frac, a.valid_frac, a.seed); report['split_method']='random'
        rejects=[]; reason_counts=Counter(); atom_stats=Counter()
        for sp, rows in split_rows.items():
            for i,row in enumerate(rows):
                try:
                    rxn, st=validate_and_convert(row, a.include_reagents, a.allow_multi_product); atom_stats.update(st)
                    rid=row.get('id') if pd.notna(row.get('id')) and row.get('id') else 'uspto_mit_%s_%06d'%(sp, len(result[sp]))
                    out={'id':rid, RXN_KEY:rxn}
                    if 'class' in row and pd.notna(row['class']): out['class']=row['class']
                    result[sp].append(out)
                except Exception as e:
                    reason=str(e); reason_counts[reason]+=1; row2=dict(row); row2['reason']=reason; rejects.append(row2)
        assert_nonempty_splits(result, a.allow_empty_splits)
        for sp in SPLITS: pd.DataFrame(result[sp], columns=['id', RXN_KEY, 'class']).dropna(axis=1, how='all').to_csv(os.path.join(a.out,'raw_%s.csv'%sp), index=False)
        pd.DataFrame(rejects).to_csv(os.path.join(a.out,'conversion_rejected.csv'), index=False)
        report.update({'rows_kept':sum(len(v) for v in result.values()), 'rows_rejected':len(rejects), 'rejection_counts_by_reason':dict(reason_counts), 'split_sizes':{k:len(v) for k,v in result.items()}, 'atom_map_statistics':dict(atom_stats)})
        json.dump(report, open(os.path.join(a.out,'conversion_report.json'),'w'), indent=2)
        if report['rows_kept']==0: raise SystemExit('No valid mapped USPTO-MIT reactions kept. Check atom maps and input format.')
    finally:
        if tmp: shutil.rmtree(tmp)
if __name__=='__main__': main()
