# DyEM: Dynamical Error Metrics 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  ![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg) ![Lightning](https://img.shields.io/badge/Lightning-2.2+-blueviolet?logo=lightning)

This repository contains the Python code and Jupyter notebooks supporting the manuscript "Dynamical Error Metrics in Machine Learning Forecasts," which introduces the use of instantaneous dynamical indices for evaluating machine learning forecasts, introducing error metrics mirroring the one commonly used in the field.

## Summary

In machine learning forecasting, standard error metrics such as mean absolute error (MAE) and mean squared error (MSE) quantify discrepancies between predictions and target values. 
However, these metrics do not directly evaluate the physical and/or dynamical consistency of forecasts, an increasingly critical concern in scientific and engineering applications.
Indeed, a fundamental yet often overlooked question is whether machine learning forecasts preserve the dynamical behavior of the underlying system. 
Addressing this issue is essential for assessing the fidelity of machine learning models and identifying potential failure modes, particularly in applications where maintaining correct dynamical behavior is crucial.
In this repository, we provide the code for reproducing the results in our paper "Dynamical Error Metrics in Machine Learning Forecasts", that can also be used to evaluate other ML learning forecasts of interests to users and practitioners.

📄 [Paper (arXiv)](https://arxiv.org/abs/2504.11074)

## 🚀 Setup
### 1. Clone the repository

```
git clone https://github.com/yourusername/project-name.git
```
### 2. Install dependencies
We recommend to use a virtual environment, such as `venv` or `conda`. To install all dependencies, use:

```
pip install -r requirements.txt
```
or
```
conda install --file requirements.txt
```

## 📊 Data
### Generate from scripts
Lorenz, KS, KF

### Download Weatehr data
ERA5:
Weatherbench2

## 🏋️‍♀️ Training
Run experiments
```
python run.py --config config/config_name.yaml --device[opt] 0 --seed[opt] 42
```

Hyperparameter optimization:

```
python sweep.py --config config/config_name.yaml --sweep_config config/config_name.yaml
```
## 📈 Evaluation
Stadard and dynamical metrics are available in 

## 📝 Citation
If you find this code or paper helpful to your work, please cite:
```bibtex
```
