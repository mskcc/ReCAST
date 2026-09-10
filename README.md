# ReCAST (Regularized Clinicogenomic Adaptive Survival Tool)

This repository contains the code for using ReCAST, a time-to-event interpretable machine learning framework originally designed to integrate clinical and genomic data for cancer prognostication.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/mskcc/ReCAST.git
cd ReCAST
```

For running ReCAST, you need to create and activate a dedicated Conda environment:

```bash
conda create -n recast python=3.10.18 pip
conda activate recast
```

Install the Python dependencies for running ReCAST:

```bash
python -m pip install -r requirements.txt
```

