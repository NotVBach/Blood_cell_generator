#!/bin/bash

# Function to check package installation
check_package() {
    package_name=$1
    command -v $package_name &> /dev/null
    if [ $? -ne 0 ]; then
        echo "$package_name is not installed. Installing..."
        if [[ $package_name == "torch" ]]; then
            pip install torch==1.4.0 torchvision==0.5.0
        elif [[ $package_name == "numpy" ]]; then
            conda install numpy=1.17 -y
        elif [[ $package_name == "scikit-learn" ]]; then
            conda install scikit-learn=0.24.1 -y
        else
            echo "No installation command defined for $package_name."
        fi
    else
        echo "$package_name is already installed."
    fi
}

# Check Python version
echo "Checking Python version..."
python_version=$(python --version)


# Check if NumPy is installed
echo "\nChecking NumPy..."
check_package "numpy"

# Check if scikit-learn is installed
echo "\nChecking scikit-learn..."
check_package "scikit-learn"

# Check if PyTorch is installed
echo "\nChecking PyTorch..."
check_package "torch"

# Check CUDA availability
echo "\nChecking CUDA availability..."
if python -c "import torch; print(torch.cuda.is_available())"; then
    echo "CUDA is available."
else
    echo "CUDA is not available."
fi