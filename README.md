# COSC2669/COSC2816 – Epilepsy Machine Learning Analysis

This repository contains the reproducible analysis used for the Case Studies in Data Science assignment.

## Research question
How effectively can machine learning identify epileptic seizure activity from EEG data across a structured research dataset and a real-world clinical dataset?

## Models
- Gradient Boosting Classifier
- Multilayer Perceptron (MLP) neural network

## Datasets
1. **BEED – Bangalore EEG Epilepsy Dataset (UCI Machine Learning Repository)**
   - Place `BEED_Data.csv` in `data/raw/`
   - Official source: https://archive.ics.uci.edu/dataset/1134/beed%3A%2Bbangalore%2Beeg%2Bepilepsy

2. **CHB-MIT Scalp EEG Database (PhysioNet)**
   - Place these files in `data/raw/`:
     - `chb01_01.edf`
     - `chb01_03.edf`
     - `chb01_04.edf`
     - `chb01_15.edf`
     - `chb01_16.edf`
     - `chb01_18.edf`
     - `chb01_21.edf`
     - `chb01_26.edf`
   - Official source: https://physionet.org/content/chbmit/1.0.0/
   - Patient annotation file: https://physionet.org/content/chbmit/1.0.0/chb01/chb01-summary.txt

Large raw datasets are intentionally not stored in this repository.

## Environment
Recommended: Python 3.11+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the analysis
1. Create/open this repository locally or in VS Code/Jupyter.
2. Put the required dataset files into `data/raw/`.
3. Open `epilepsy_analysis.ipynb`.
4. Select **Restart Kernel and Run All**.
5. Check the final tables and figures.
6. The notebook writes output figures and a metrics CSV to `results/`.

## Evaluation design
### BEED
- Original four-class target is retained.
- 5-fold stratified cross-validation.
- Macro precision, recall, F1 and one-vs-rest ROC-AUC are reported.

### CHB-MIT
- Seven seizure-containing recordings from patient `chb01`.
- Official seizure start/end annotations from PhysioNet.
- 5-second windows.
- Equal numbers of seizure and non-seizure windows.
- Non-seizure windows are selected at least 60 seconds away from annotated seizures.
- 26 aggregate time/frequency-domain EEG features are extracted.
- Leave-one-recording-out evaluation is used to reduce leakage between windows from the same seizure episode.

## Important limitations
- The downloadable BEED table does not provide participant identifiers, so participant-level splitting cannot be performed.
- The CHB-MIT experiment uses one patient only, so results are not evidence of generalisation to unseen patients.
- The CHB-MIT modelling sample is artificially balanced and does not reflect real clinical seizure prevalence.
- Results are for an academic case study, not a clinically validated seizure-detection system.
