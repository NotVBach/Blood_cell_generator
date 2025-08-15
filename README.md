## Requirement
- Python 3.12
- GPU (Optional)

## Set up 1: using pip
- NumPy version: 2.3.1
- PyTorch version: 2.7.1+cu126
- scikit-learn version: 1.7.1
- CUDA version: 12.6

## Set up 2: using conda
`conda env create -f environment.yml`

## How to run
- `python main.py --method deep_smote --mode train`
- `python main.py --method deep_smote --mode generate`
- `python main.py --method balance_gan --mode train`
- `python main.py --method balance_gan --mode generate`
