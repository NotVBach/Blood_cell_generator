# MNIST config

# args = {
#     'dim_h': 64,          # factor controlling size of hidden layers
#     'n_channel': 1,       # number of channels in the input data
#     'n_z': 300,           # number of dimensions in latent space
#     'sigma': 1.0,         # variance in n_z
#     'lambda': 0.01,       # hyper param for weight of discriminator loss
#     'lr': 0.0002,         # learning rate for Adam optimizer
#     'epochs': 2,        # number of epochs to run for
#     'batch_size': 100,    # batch size for SGD
#     'save': True,         # save weights at each epoch of training if True
#     'train': True,        # train networks if True, else load networks
#     'dataset': 'mnist'    # specify which dataset to use
# }


# noaug config

args = {
    'dim_h': 64,            # factor controlling size of hidden layers
    'n_channel': 3,         # number of channels (3 for RGB)
    'n_z': 300,             # number of dimensions in latent space
    'sigma': 1.0,           # variance in n_z
    'lambda': 0.01,         # hyper param for weight of discriminator loss
    'lr': 0.0002,           # learning rate for Adam optimizer
    'epochs': 2,          # how many epochs to run for training
    'batch_size': 100,      # batch size for SGD
    'save': True,           # save weights at each epoch if True
    'train': True,          # train networks if True, else load networks
    'dataset': 'noaug',     # specify the dataset
    'image_size': 603,       # resize images to this size
    'num_classes': 8        # number of defect classes (1–8)
}