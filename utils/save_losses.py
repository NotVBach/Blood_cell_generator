import os
import pandas as pd

def save_losses(loss_dict, output_dir, method, epoch):
    """
    Save training losses to a CSV file.
    
    Args:
        loss_dict: Dict with loss names and values (e.g., {'train_loss': 0.1, 'mse': 0.05}).
        output_dir: Directory to save the CSV file.
        method: 'deep_smote' or 'balance_gan'.
        epoch: Current epoch number.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{method}_training_losses.csv')
    
    # Convert loss_dict to DataFrame
    loss_dict['epoch'] = epoch
    df = pd.DataFrame([loss_dict])
    
    # Append to CSV or create new
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)