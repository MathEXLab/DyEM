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
git clone https://github.com/MathEXLab/DyEM.git
```
### 2. Install dependencies
We recommend to use a virtual environment, such as `venv` or `conda` to manage the packages. To install all dependencies, use:

```
pip install -r requirements.txt
```
or
```
conda install --file requirements.txt
```

## 📊 Data
### Generate from scripts
#### Lorenz dataset
The Lorenz 63 dataset can be generated with
```
python /data/lorenz/data_generation.py --nt 150000
```
After that, preprocess the data with
```
python /data/lorenz/data_process.py
```

#### Kuramoto-Sivashinsky (KS) dataset
The code for generating KS dataset is adopted from the github repository [pyks](https://github.com/jswhit/pyks).
In this repo, generate data with
```
python /data/ks/ks_gen.py
```
After that, preprocess the data with
```
python /data/ks/data_process.py
```


#### Kolmogorov Flow (KF) dataset
We use the code provided by [KolSol](https://github.com/MagriLab/KolSol) to generate KF dataset, with the parameter
```
--resolution 64 --re 14.4 --time-simulation 10000 --nf 2
```

After the data is generated, calculate vorticity and downsample data with:
```
python /data/kf/preprocess.py --truncate_before 10000
```

### Download Weatehr data
ERA5:
Weatherbench2

## 🏋️‍♀️ Training
Run experiments
```
python run.py --config config/config_name.yaml --device[opt] 0 --seed[opt] 42
```

Run hyperparameter optimization:

```
python sweep.py --config config/config_name.yaml --sweep_config config/config_name.yaml
```
## 📈 Evaluation
Stadard and dynamical metrics are available in `src/utils/dy_metrics.py`.

## 📝 Citation
If you find this code or paper helpful to your work, please cite:
```bibtex
@article{fang2025dynamical,
  title={Dynamical errors in machine learning forecasts},
  author={Fang, Zhou and Mengaldo, Gianmarco},
  journal={arXiv preprint arXiv:2504.11074},
  year={2025}
}

```
