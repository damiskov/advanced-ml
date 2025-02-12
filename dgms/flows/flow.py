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


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='models/model.pt', choices=['models/model_tg.pt', 'models/model_cb.pt'], help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='figs/samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    train_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    D = next(iter(train_loader)).shape[1]
    base = GaussianBase(D)

    # Define transformations
    transformations =[]
    mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
    
    num_transformations = 5
    num_hidden = 8

    # Make a mask that is 1 for the first half of the features and 0 for the second half
    mask = torch.zeros((D,))
    mask[D//2:] = 1
    
    for i in range(num_transformations):
        mask = (1-mask) # Flip the mask
        # scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        # scale_net = BasicScaleNet(D, num_hidden)
        scale_net = DeepScaleNet(D, 3*num_hidden, 3*2) # Same number of hidden units, double number of layers
        #translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        # translation_net = BasicTranslationNet(D, num_hidden)
        translation_net = DeepTranslationNet(D, 3*num_hidden, 3*2) # Same number of hidden units, double number of layers
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    # Define flow model
    model = Flow(base, transformations).to(args.device)

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

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,))).cpu() 

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
        plt.savefig(args.samples)
        plt.close()