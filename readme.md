# Captcha Reader Model

A simple model for reading captcha images

## Install Requirements

Use the package manager pip to install Requirements.

```bash
pip install -r requirements.txt
```
Requirements:\
keras 2.9.0\
matplotlib 3.5.3\
numpy 1.23.1\
pandas 1.4.3\
tensorflow 2.9.1

## Training
For training the model run following command

```bash
python training.py --batch_size --n_epochs
```
For example:\
Training with batch size of 32 for 50 epochs

```bash
python training.py 32 50 
```

## Inference
For inference with your desired image

```bash
python inference.py /path/to/image 
```
Example:
```bash
python inference.py ./samples/2enf4.png
```
