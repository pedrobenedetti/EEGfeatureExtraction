# EEGfeatureExtraction
Pipeline for extracting features from EEG signals and performing subsequent analysis. The scripts are designed for signals acquired with a 128-channel BioSemi system. After feature extraction, the resulting data are reshaped, normalized, and their dimensionality is reduced using PCA.

---

## General pipeline
Describí en 3-6 líneas cuál es la lógica general del flujo de trabajo.

### 1. `featureExtraction_1.py`

**Purpose**  
This script receives a set of signals from multiple recordings, listed in the `file_all` list in the main block. All files are assumed to be located in the same folder, whose path is specified in `path`. Features are extracted for the conditions listed in `conds` and for the frequency bands defined in the `BANDS` dictionary, which contains both the band names and their corresponding frequency ranges.

The extracted features are periodic band power, aperiodic exponent and offset (modeled using the FOOOF algorithm), weighted phase lag index (wPLI), weighted symbolic mutual information (wSMI), Lempel-Ziv complexity (LZC), transfer entropy (TE), and permutation entropy (PE).
**Inputs**
- Input 1
- Input 2

**Outputs**
- Output 1
- Output 2

**Notes**  
Important clarifications, assumptions, methodological details, etc.

---

## Repository structure

```text
EEGfeatureExtraction/
├── featureExtraction_1.py
├── featureExtraction_Reshape_2.py
├── featureExtraction_Normalize_3.py
├── featureExtraction_PCA_4.py
└── README.md
