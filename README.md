# Detection of Peripheral Blood Cell Images using Deep Learning

## Some method that I might do
- Analyse blood cells dataset: Examine image quality, gain insight of the dataset through data visualization methods.
- Address imbalance dataset: Apply data augmentation methods, generative image technique Synthetic Minority Over-sampling Technique (SMOTE), Generative - Adversarial Networks (GANs).
- Deep learning method: Detect and classify blood cells using Convolutional Neural Networks (CNNs) such as YOLO, VGG19,... or some attention mechanism like Vision transformer.
- Evaluation and Validation: Evaluating models performance using metrics such as accuracy, precision, recall, F1-score, …


## Objective that I should done
- Identify and quantify class imbalances.
- Implement basic data augmentation and generative image technique.
- Train functional CNN models then compare them with evaluated metrics.


# Difficulties
- Image has bounding boxes, can bbox be generated via smote
- One image can have 2 cells, how to deal with that.
- Convert to Json if have bounding box

## Requirement
- Python 3.6
- 

ssh bach@192.168.0.11

## set up
conda create -n smote python=3.6
conda activate smote
conda install cudatoolkit=10.1 -c anaconda -y
nvidia-smi
pip install torch==1.4.0 torchvision==0.5.0
conda install scikit-learn=0.24.1 numpy=1.17
python -c"import pytorch; torch.cuda.is_available()"

