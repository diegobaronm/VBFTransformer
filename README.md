# VBF Transformer Project

This project implements a Transformer-based model for VBF Di-Tau analysis.

## Installation

### 1. Create the Conda Environment

First, create the conda environment using the provided `requirements.yaml` file:

```bash
conda env create -f requirements.yaml
```

Then, activate the environment:

```bash
conda activate vbf_ditau
```

### 2. Install the VBFTransformer Package

Install the project as a Python package in editable mode. This allows you to make changes to the source code and have them immediately reflected without reinstalling.

```bash
pip install -e .
```

## Usage

The main script `VBFTransformer.py` is controlled via the command line and configured with Hydra. You can run the project in three different modes: `train`, `predict`, and `performance`.

### Training the Model

To train the model, run:

```bash
python VBFTransformer.py --config-name config opts.mode=train
```

### Generating Predictions

To generate predictions with a trained model, run:

```bash
python VBFTransformer.py --config-name config opts.mode=predict
```

### Evaluating Performance

To evaluate the model's performance (e.g., generate ROC curves and confusion matrices), run:

```bash
python VBFTransformer.py --config-name config opts.mode=performance
```

You can customize the behavior by modifying the `configs/config.yaml` file or by overriding parameters from the command line. For example:

```bash
python VBFTransformer.py opts.mode=train model.learning_rate=0.002
```
