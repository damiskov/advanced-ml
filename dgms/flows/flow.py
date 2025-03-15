# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.3 (2024-02-11)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb
# - https://github.com/VincentStimper/normalizing-flows/tree/master

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
from utils.scale_nets import (
    BasicScaleNet,
    DeepScaleNet,
)
from utils.translation_nets import (
    BasicTranslationNet,
    DeepTranslationNet,
)
from utils.base_distributions import GaussianBase
from utils.flow_models import Flow
from utils.coupling_layers import MaskedCouplingLayer
from utils.training import train
from pytorch_model_summary import summary

def build_flow_model(masks, network_type, D, num_hidden):
    """
    Build a flow model with the given masks and network type.

    Parameters:
    masks: [List[torch.Tensor]]
        List of masks for each coupling layer.
    network_type: [str]
        Type of networks to use {basic, enhanced}.
    D: [int]
        Dimension of the input data.
    num_hidden: [int]
        Number of hidden units in the networks.

    Returns:
    model: [Flow]
        A flow model with the given masks and networks.
    """
    print("Building flow model")
    transformations = []
    for i, mask in enumerate(masks):
        
        if network_type == 'basic':
            scale_net = BasicScaleNet(D, num_hidden)
            translation_net = BasicTranslationNet(D, num_hidden)
        
        elif network_type == 'enhanced':

            scale_net = DeepScaleNet(D, 4*num_hidden, 12)
            translation_net = DeepTranslationNet(D, 4*num_hidden, 12)
        
        else:
            raise ValueError(f'Invalid network type: {network_type}')

        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    model = Flow(base, transformations).to(args.device)

    # Print a summary
    print('----------------- Built Model -----------------')
    print(summary(model, torch.zeros((1, D)), show_input=False, show_hierarchical=False))
    return model

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='models/model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='figs/samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')


    # Some custom args
    # Flag to change which dataset to use
    parser.add_argument('--dataset', type=str, default='toy', choices=['mnist', 'toy'], help='dataset to use {mnist, toy} (default: %(default)s)')
    # Flag to change which scaling + translation networks to use
    parser.add_argument('--networks', type=str, default='basic', choices=['basic', 'enhanced'], help='type of networks to use {basic, enhanced} (default: %(default)s)')
    # Flag to change which masking pattern to use
    parser.add_argument('--mask', type=str, default='basic', choices=['checkerboard', 'random', 'basic'], help='masking pattern to use {checkerboard, random} (default: %(default)s)')


    args = parser.parse_args()

    #  define a unique model name based on args
    model_name = f'models/model_networks={args.networks}_mask={args.mask}_dataset={args.dataset}'
    if args.dataset=='mnist': model_name += '.pt'
    else: model_name += f'_data={args.data}.pt'

    # We also do the same for the sampling file
    sample_dir = f'figs/samples_networks={args.networks}_mask={args.mask}_dataset={args.dataset}'
    if args.dataset == 'mnist': sample_dir += '.png'
    else: sample_dir += f'_data={args.data}.png'

    print(f'Model name: {model_name}')
    args.model = model_name
    print(f'Sample dir: {sample_dir}')
    args.samples = sample_dir


    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Some hyperparameters for tuning networks

    num_transformations = 10 # Increased for MNIST
    num_hidden = 8
    transformations = []

    # Load/Generate the data
    # Also change masking pattern depending on the dataset

    match args.dataset:
        case 'mnist':
            # Load MNIST data
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                                transforms.Lambda(lambda x: x.flatten())
                                ])),
                                batch_size=args.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data/', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                                transforms.Lambda(lambda x: x.flatten())
                                ])),
                                batch_size=args.batch_size, shuffle=True
            )
            # Convert to device
            train_loader = [(x.to(args.device)) for x, _ in train_loader]
            test_loader = [(x.to(args.device)) for x, _ in test_loader]
            # Define prior distribution
            # NOTE: Fixed for all datasets
            D = 28*28
            base = GaussianBase(D)

            # We use either a random initialised mask, or the 'checkerboard' mask
            # TODO: Implement different masking patterns

            match args.mask:
                case 'random':
                    # Creating a sequence of random masks
                    mask_lst = [torch.randint(0, 2, (D,)) for _ in range(num_transformations)]
                case 'checkerboard':
                    # Not really sure if this is correct?
                    mask_lst = []
                    mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
                    for _ in range(num_transformations):
                        mask = (1-mask)
                        mask_lst.append(mask)
                case _:
                    raise NotImplementedError(f'Mask {args.mask} not implemented for dataset {args.dataset}')


        case 'toy':
            
            # Generate toy data
            n_data = 10000000
            toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
            train_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
            
            # Define prior distribution
            # NOTE: Fixed for all datasets
            D = next(iter(train_loader)).shape[1]
            base = GaussianBase(D)

            # We only use a basic mask here
            # Make a mask that is 1 for the first half of the features and 0 for the second half
            mask = torch.zeros((D,))
            mask[D//2:] = 1

            # Define mask for each coupling layer - Should be flipped for each layer
            mask_lst = []
            for i in range(num_transformations):
                mask = (1-mask)
                mask_lst.append(mask)

        case _:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    
    # # Make a mask that is 1 for the first half of the features and 0 for the second half
    # mask = torch.zeros((D,))
    # mask[D//2:] = 1
    
    # for i in range(num_transformations):
    #     mask = (1-mask) # Flip the mask
    #     # scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
    #     # scale_net = BasicScaleNet(D, num_hidden)
    #     scale_net = DeepScaleNet(D, 3*num_hidden, 3*2) # Same number of hidden units, double number of layers
    #     #translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
    #     # translation_net = BasicTranslationNet(D, num_hidden)
    #     translation_net = DeepTranslationNet(D, 3*num_hidden, 3*2) # Same number of hidden units, double number of layers
    #     transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))


    # # Define flow model
    # model = Flow(base, transformations).to(args.device)

    model = build_flow_model(mask_lst, args.networks, D, num_hidden)


    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)


    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        print('------------ Loaded Model ------------')
        print(summary(model, torch.zeros((1, D)), show_input=False, show_hierarchical=False))




        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,))).cpu() 

        def plot_samples(samples, dataset):
            
            match dataset:
                case 'toy':
                    # Plot the density of the toy data and the model samples
                    coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
                    prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

                    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                    im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
                    ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
                    ax.set_xlim(toy.xlim)
                    ax.set_ylim(toy.ylim)
                    ax.set_aspect('equal')
                    fig.colorbar(im)
                    print(f"saved samples to {args.samples}")
                    plt.savefig(args.samples)
                    plt.close()
                
                case 'mnist':
                    # Plot the samples
                    # Only take 10 samples
                    if samples.shape[0] > 10: samples = samples[:10]
                    samples = samples.view(-1, 1, 28, 28)
                    save_image(samples, args.samples, nrow=10)



                
                case _:
                    raise NotImplementedError(f'Dataset {dataset} not implemented')

        
       
        plot_samples(samples, args.dataset)
        # # Plot the density of the toy data and the model samples
        # coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        # prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        # im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        # ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        # ax.set_xlim(toy.xlim)
        # ax.set_ylim(toy.ylim)
        # ax.set_aspect('equal')
        # fig.colorbar(im)
        # print(f"saved samples to {args.samples}")
        # plt.savefig(args.samples)
        # plt.close()