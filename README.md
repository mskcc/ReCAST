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

## Instructions

For running ReCAST we simplified it to resemble the general scikit-survival use.

We recommend using a Jupyter notebook in this first Python implementation. For using the required packages, select as kernel the "recast" conda environment you created before

First, your notebook need to have access to the repository where you copied the github repository. Add the following command inside your Jupyter notebook:

```python 
import sys
sys.path.append('/path_to/ReCAST/code') 
from ReCAST import ReCAST
```

### Fit in the training cohort

Second, initialize the model. 

*In the default settings, the model will contain n=100 base learners, each one trained on a bootstrapped cohort, and the regularization strenght of each model will apply an adaptive penalty to maximize the model sensitivity for rare/sparse yet meaningful predictors.*

```python
model = ReCAST()
```

Third, fit the model. 
You have to provide:

i) the dataframe containig the variables (X1, X2, X3 etc), using variables columns, and patients as indexes. 

ii) a dataframe containg the 'Time' variable and 'Event' variable. Please rename the columns to have these names

```python
model.fit(dataframe_variables, dataframe_timetoevent)
```

You can also easily see what is the Harrel's C-index for the training samples:

```python
model.calculate_train_cindex(dataframe_timetoevent)
```
*The model was designed to compute out-of-bag (if bootstrap=True) or out-of-fold (if bootstrap=False during model initialization) to make unbiased training risk scores, which we assumed is critical during early model development*

And see the feature importance in a easy way:
```python
model.plot_selection_frequency()
```

For clinical decision making, is fundamental to have **risk groups**. ReCAST was designed to implement a Gaussian Mixture Model to overcome the bias of using arbitrary quantile scores. Currently ReCAST supports either n=2 risk groups or n=3 risk groups. 
For defining risk-groups:

```python
model.define_2_risk_groups()

#or if you assume in your clinical scenario the rationale is to have n=3 risk groups:
model.define_3_risk_groups()

#you can also easily see the distribution of the risk categories (density components) simply using the plot=True
model.define_2_risk_groups(plot=True)
```

Before going to the test, you can get the risk-scores of your train set or get the risk-group assignment (before, run the function above for defining the risk groups) for you training set in an easy way:

```python 
#for risk scores
model.risk_scores_train

#for risk groups
model.get_train_risk_groups_2()

#or if you selected n=3 risk groups
model.get_train_risk_groups_3()
```

### Prediction in independent samples

Fourth, for **prediction** to independent samples, you can either get the continuous risk-scores, or the risk-class assignment (where ReCAST will use the previously defined cutoffs on the training set). 

*As input, just provide the dataframe with the variabes of your test samples; no need to provide here the dataframe time-to-event of the test samples. ReCAST will authomatically consider just the variables included in your training samples*

```python
#for continuous risk-scores
model.predict(dataframe_variables_test)

#for risk-group assignment in the test
model.predict(dataframe_variables_test, classify=True, groups = 3) #or groups = 2 if you selected before 2 groups
```

All the functions above, starting from model initialization, plotting and so on have different possibile settings including additional functions not listed above for simplicity, each described in detail in /ReCAST/code


