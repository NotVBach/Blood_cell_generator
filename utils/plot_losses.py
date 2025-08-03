import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_losses(csv_path, output_dir, method):
    """
    Plot training losses from a CSV file and save the plot.
    
    Args:
        csv_path: Path to the CSV file with losses.
        output_dir: Directory to save the plot.
        method: 'deep_smote' or 'balance_gan'.
    """
    if not os.path.exists(csv_path):
        print(f"No CSV file found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for column in df.columns:
        if column != 'epoch':
            plt.plot(df['epoch'], df[column], label=column)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{method.capitalize()} Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{method}_loss_plot.png'))
    plt.close()