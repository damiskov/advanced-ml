# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

# NOTE: Heavily modified and partitioned into separate files for better modularity and readability.
# David Miles-Skov 2025-02-07

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 

# Models
from models.vaes import (
    VAE,
    VAE_MoG,
    VAE_gaussian_output,
    VAE_CNN,
)
from models.decoders import (
    BernoulliDecoder,
    GaussianDecoder,
    BernoulliDecoderCNN,
)
from models.encoders import (
    GaussianEncoder,
)
from models.priors import (
    GaussianPrior,
    MixtureGaussianPrior,
)
# utils
from utils.training import (
    train,
)
from utils.evaluating import (
    evaluate_elbo,
    visualize_latent_space_2d,
    visualize_latent_space_with_pca,
)

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'all'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='gaussian_prior', choices=['gaussian_prior', 'MoG_prior', 'gaussian_output', 'cnn_encoder_decoder'], help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, metavar='N', help='number of components in the mixture Gaussian prior (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    

    # Latent dimension
    M = args.latent_dim
    
    # Match case to build model
    match args.model:
        
        case 'gaussian_prior':
            # Load MNIST as binarized at 'threshold' and create data loaders
            threshold = 0.5
            mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/binary_mnist/', train=True, download=True,
                                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/binary_mnist/', train=False, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            # Define encoder and decoder networks
            encoder_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512), # Input is 28x28 = 784 (flattened image)
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, M*2), # Output is 2M (mean and std of the Gaussian distribution). M = latent dimension
            )

            decoder_net = nn.Sequential(
                nn.Linear(M, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Unflatten(-1, (28, 28))
            )

            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            prior = GaussianPrior(M)
            model = VAE(prior, decoder, encoder).to(device)
     
        
        case 'MoG_prior':
            # Load MNIST as binarized at 'threshold' and create data loaders
            threshold = 0.5
            mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/binary_mnist/', train=True, download=True,
                                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/binary_mnist/', train=False, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            # Define encoder and decoder networks
            encoder_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512), # Input is 28x28 = 784 (flattened image)
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, M*2), # Output is 2M (mean and std of the Gaussian distribution). M = latent dimension
            )

            decoder_net = nn.Sequential(
                nn.Linear(M, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Unflatten(-1, (28, 28))
            )
            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            prior = MixtureGaussianPrior(M, args.num_components)
            model = VAE_MoG(prior, decoder, encoder).to(device)
            
        
        case 'gaussian_output':
            # Load MNIST as continuous and create data loaders
            mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/continuous_mnist', train=True, download=True,
                                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/continuous_mnist', train=False, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float().squeeze())])),
                                                            batch_size=args.batch_size, shuffle=True)
            # Define encoder and decoder networks
            encoder_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512), # Input is 28x28 = 784 (flattened image)
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, M*2), # Output is 2M (mean and std of the Gaussian distribution). M = latent dimension
            )

            decoder_net = nn.Sequential(
                nn.Linear(M, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 784*2), # Note the output is 2*784 (mean and log-variance). Differs from the log-variance output in the GaussianDecoder
                nn.Unflatten(-1, (2, 28, 28)) # Modified to output 2 channels (mean and log-variance)
            )
            
            encoder = GaussianEncoder(encoder_net)
            decoder = GaussianDecoder(decoder_net, M)
            prior = GaussianPrior(M)
            model = VAE_gaussian_output(prior, decoder, encoder).to(device) # Since we are using a simple Gaussian prior we can use the basic VAE model
            
        
        case 'cnn_encoder':
            # Load MNIST as binarized at 'threshold' and create data loaders
            threshold = 0.5
            mnist_train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    'data/binary_mnist/', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),  
                        transforms.Lambda(lambda x: (x > threshold).float())  # Add missing channel dimension
                    ])
                ),
                batch_size=32, shuffle=True
            )

            mnist_test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    'data/binary_mnist/', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x > threshold).float())  # Add missing channel dimension
                    ])
                ),
                batch_size=32, shuffle=True
            )

            # Define encoder and decoder networks
            encoder_net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (batch, 32, 14, 14)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (batch, 64, 7, 7)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * M)  # Output mean and std for the latent space
            )
            decoder_net = nn.Sequential(
                nn.Linear(M, 256),
                nn.ReLU(),
                nn.Linear(256, 64 * 7 * 7),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),  # Reshape to match the CNN input shape
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (batch, 32, 14, 14)
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (batch, 1, 28, 28)
            )


            prior = GaussianPrior(M) # Using a simple Gaussian prior for closed-form KL divergence
            encoder = GaussianEncoder(encoder_net)
            decoder = BernoulliDecoderCNN(decoder_net)
            model = VAE(prior, decoder, encoder).to(device)
        
    # Run model
    match args.mode:
        case 'train':
            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train model
            losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device)

            # Save model
            torch.save(model.state_dict(), f"{args.model}.pt")
        
        case 'sample':
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), f'figs/{args.model}_samples.png')
        
        case 'eval':
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
            avg_elbo = evaluate_elbo(model, mnist_test_loader, args.device)
            visualize_latent_space_2d(model, mnist_test_loader, args.device)
            visualize_latent_space_with_pca(model, mnist_test_loader, args.device)
        
        case 'all':

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Train model
            losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device)
            # Save model
            torch.save(model.state_dict(), f"{args.model}.pt")
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), f'figs/{model.name}_samples.png')
            # Evaluate model
            avg_elbo = evaluate_elbo(model, mnist_test_loader, args.device)
            # Visualize latent space
            visualize_latent_space_2d(model, mnist_test_loader, args.device)
            visualize_latent_space_with_pca(model, mnist_test_loader, args.device)