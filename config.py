args = {
    'dim_h': 64,                # factor controlling size of hidden layers
    'n_channel': 3,             # number of channels
    'n_z': 300,                 # number of dimensions in latent space
    'bbox_dim': 4,              # bounding box dimension
    'sigma': 1.0,               # variance in n_z
    'lambda': 0.01,             # hyper param for weight of discriminator loss
    'bbox_lambda': 0.1,         # bounding box hyper param
    'lr': 0.0002,               # learning rate for Adam optimizer
    'epochs': 200,              # how many epochs to run for training
    'batch_size': 100,          # batch size for SGD
    'save': True,               # save weights at each epoch if True
    'train': True,              # train networks if True, else load networks
    'dataset': 'defect',        # specify the dataset
    'base_dir': 'noaug',        # path to dataset
    'image_size': (128, 128)    # resize images to this size
}