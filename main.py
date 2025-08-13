import argparse
from config import args
import os

def main():
    parser = argparse.ArgumentParser(description='Run Deep SMOTE or Balance GAN')
    parser.add_argument('--method', type=str, choices=['deep_smote', 'balance_gan'], default=args['method'],
                        help='Method to run: deep_smote or balance_gan')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train',
                        help='Mode: train or generate')
    parser.add_argument('--image_size', type=int, nargs=2, default=args['image_size'],
                        help='Image size as height width (e.g., 32 32)')
    args_cmd = parser.parse_args()

    args['method'] = args_cmd.method
    args['train'] = args_cmd.mode == 'train'
    args['image_size'] = tuple(args_cmd.image_size)
    args['output_dir'] = os.path.join('synthetic', args['method'])

    if args['method'] == 'deep_smote':
        if args['train']:
            os.system('python deep_smote/train.py')
        else:
            os.system('python deep_smote/generate.py')
    elif args['method'] == 'balance_gan':
        if args['train']:
            os.system('python balance_gan/train.py')
        else:
            os.system('python balance_gan/generate.py')
    else:
        raise ValueError(f"Unknown method: {args['method']}")

if __name__ == '__main__':
    main()