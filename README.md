[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7837349.svg)](https://doi.org/10.5281/zenodo.7837349)
# Retrosynthesis prediction using an end-to-end graph neural network for molecular graph editing
Inspired by the arrow-pushing formalism in chemical reaction mechanisms, we present a novel end-to-end architecture for retrosynthesis prediction, Graph2Edits, based on graph neural network to predict the edits of the product graph in an auto-regressive manner, and sequentially generates transformation intermediates and final reactants according to the predicted edits sequence. 
## Environment Requirements  
Create a virtual environment to run the code of Graph2Edits.
Install pytorch with the cuda version that fits your device.
```
conda create -n graph2edits python=3.7 \
conda activate graph2edits \
conda install -c conda-forge rdkit=2019.09.2 \
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch \
pip install numpy==1.17.3 \  
```
## Data preprocessing
1) generate the edit labels and the edits sequence for reaction
```
python preprocess.py --mode train \
python preprocess.py --mode valid \
python preprocess.py --mode test \
```
2) prepare the data for training
```
python prepare_data.py
```
## Train Graph2Edits model
Go to the graph2edits folder and run the following to train the model with specified dataset (default: USPTO_50k)
```
python train.py --dataset uspto_50k --use_rxn_class False
```
The trained model will be saved at graph2edits/experiments/uspto_50k/without_rxn_class/
## Evaluate using a trained model
To evaluate the trained model, run
```
python eval.py
```
to get the raw prediction file saved at graph2edits/experiments/.../pred_results.txt
## Reproducing our results
To reproduce our results, run
```
python eval.py --dataset uspto_50k --use_rxn_class False or True --experiments 27-06-2022--10-27-22 or 30-06-2022--00-19-29
```
This will display the results for reaction class unknown and known setting

## Running Graph2Edits on USPTO-MIT / USPTO-480k

USPTO-MIT is supported as dataset name `uspto_mit`. The converter accepts mapped reaction SMILES in Jin/WLN text files (first whitespace token), DeepChem CSV files with a `reactions` column, or existing Graph2Edits CSV files with `reactants>reagents>production`. Atom maps are required: product atoms must have nonzero unique atom-map numbers and every product atom map must appear on the reactant side. The workflow does not add atom maps and `uspto_mit` has no default reaction-class labels, so do not use `--use_rxn_class`.

Convert raw data:
```bash
python scripts/convert_uspto_mit.py \
  --input data/raw_uspto_mit \
  --format auto \
  --out data/uspto_mit \
  --include_reagents false \
  --split predefined
```
This writes `data/uspto_mit/raw_train.csv`, `raw_valid.csv`, `raw_test.csv`, plus `conversion_report.json` and `conversion_rejected.csv`.

Canonicalize products while preserving map consistency:
```bash
python canonicalize_prod.py --dataset uspto_mit --mode train
python canonicalize_prod.py --dataset uspto_mit --mode valid
python canonicalize_prod.py --dataset uspto_mit --mode test
```
Outputs are `canonicalized_train.csv`, `canonicalized_valid.csv`, and `canonicalized_test.csv` with per-split rejection reports.

Preprocess edit labels; `auto` prefers canonicalized files when present:
```bash
python preprocess.py --dataset uspto_mit --mode train --input_stage auto --lg_min_count 50
python preprocess.py --dataset uspto_mit --mode valid --input_stage auto
python preprocess.py --dataset uspto_mit --mode test --input_stage auto
```
Training vocabularies are saved under `data/uspto_mit/train/`, and valid/test coverage is reported against the train vocabularies.

Prepare tensor batches:
```bash
python prepare_data.py --dataset uspto_mit --mode train --batch_size 256 --max_steps 9
python prepare_data.py --dataset uspto_mit --mode valid --batch_size 256 --max_steps 9
python prepare_data.py --dataset uspto_mit --mode test --batch_size 256 --max_steps 9
```
Batches are written to `data/uspto_mit/{train,valid,test}/without_rxn_class/`.

Train (a GPU is expected for full 480k training):
```bash
python train.py --dataset uspto_mit --epochs 150 --lr 0.0001 --num_workers 6 --print_every 200
```
Experiments are saved under `experiments/uspto_mit/without_rxn_class/<timestamp>/`. Each run writes normal `epoch_N.pt` checkpoints and maintains `best.pt` and `latest.pt`.

Evaluate:
```bash
python eval.py --dataset uspto_mit --experiments <timestamp> --checkpoint best --beam_size 10 --max_steps 9
```
Predictions are written under the experiment directory as `pred_results.txt` or `pred_results_1.txt`, etc. if the previous file exists.

Common failure modes include missing product atom maps, duplicate atom maps, product atom maps absent from reactants, invalid RDKit SMILES, multiple products when `--allow_multi_product false`, edit sequences longer than `--max_steps`, unseen valid/test edit labels, and attempting to use reaction-class labels with USPTO-MIT.

For the complete conversion-through-training flow, use:
```bash
bash scripts/run_uspto_mit.sh data/raw_uspto_mit data/uspto_mit
```
