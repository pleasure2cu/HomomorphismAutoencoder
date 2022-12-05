from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary

from networks import SmallDensePhi, SmallConvEncoder, SmallConvDecoder
from data_creation import get_u_rotation_dataset
from homomorphism_autoencoder import HomomorphismAutoencoder, hae_loss


def get_a_generated_dataset(grid_size, nbr_samples, nbr_steps) -> Tuple[np.ndarray, np.ndarray]:
    observations, actions = get_u_rotation_dataset(
        u_side_length=6,
        grid_side_length=grid_size,
        nbr_steps=nbr_steps,
        nbr_samples=nbr_samples
    )
    return observations, actions


def main():
    """
    The following code is meant to show how to use the homomorphism autoencoder. Part of that is also to use few epochs,
    a small dataset, and small networks so that it can easily also be run on a laptop. The performance of the model is
    accordingly bad. So don't be alarmed if you see that the reconstruction is not very good.
    Only be concerned if you get errors :)
    """

    # some parameters are set here for better readability
    latent_dim = 2  # the dimensionality of our latent space
    grid_size = 8  # the side length of the grids we want to work on
    gamma = 40.0  # from the paper (gamma is the weight for the prediction part of the loss)
    nbr_epochs = 10
    show_plots = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get a dataset to train on. In this example we generate images of a "U" that get rotated by multiples of 90 degrees
    # this here is only for demonstration purposes and the user is encouraged to use their own dataset
    observations, actions = get_a_generated_dataset(grid_size=grid_size, nbr_samples=100, nbr_steps=1)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # to create the HAE, we need 3 ingredients:
    #   1. an encoder that maps the input image to a latent representation
    #   2. a decoder that maps the latent representation to an image
    #   3. a network (called phi) that maps the actions to values we can then use to compute the rho(g) from the paper
    # here we use small example networks, the user is encouraged to use their own
    encoder = SmallConvEncoder(
        nbr_channels_start=1, kernel_sizes=[3, 3], channels=[16, 32], strides=[1, 1],  # these values parametrize the conv layers
        dense_widths=[4*4*32, 128, latent_dim]  # these values parametrize the dense layers after the conv layers
    )
    summary(encoder, (1, grid_size, grid_size))  # print a summary of the encoder to the command line
    decoder = SmallConvDecoder(
        kernel_sizes=[3, 3, 1], channels=[16, 16, 1], strides=[1, 1, 1],  # these values parametrize the transpose conv layers
        dense_widths=[latent_dim, 128, 4*4*32],  # these values parametrize the dense layers before the transpose conv layers
        shape_after_dense=(32, 4, 4),  # this shape is used to reshape the output of the dense layers before the transpose conv layers
        remove_channel_at_end=True  # this is used to remove the channel dimension at the end of the decoder
    )
    summary(decoder, (latent_dim,))  # print a summary of the decoder to the command line
    phi = SmallDensePhi([1, 32, 32, 4])

    hae = HomomorphismAutoencoder(
        encoder, decoder, phi,
        (1, grid_size, grid_size),  # the shape of the input images
        (grid_size, grid_size),  # the shape of the output images (we are working on black and white images, so there is no need for a channel)
        latent_dim,
        block_side_lengths=[2]
    ).to(device)

    # get the optimizer
    optimizer = torch.optim.Adam(hae.parameters())

    # train model
    for epoch in range(nbr_epochs):
        print(f"Epoch {epoch+1}/{nbr_epochs}")
        total_loss = 0
        total_pred_loss = 0
        total_recon_loss = 0
        for batch_idx, (obs, acts) in enumerate(dataloader):
            obs = obs.to(device)
            acts = acts.to(device)
            optimizer.zero_grad()

            latent_obs, rho, recon_0_steps, latent_t_steps, recon_t_steps = hae(obs.unsqueeze(2), acts)
            loss, recon_loss, weighted_pred_loss = hae_loss(
                obs, latent_obs, recon_0_steps, recon_t_steps, latent_t_steps, t_skip=1, gamma=gamma
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += weighted_pred_loss.item()
            total_recon_loss += recon_loss.item()

            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"Loss: {total_loss} (recon {total_recon_loss} + pred {total_pred_loss})")
                total_loss = 0
                total_pred_loss = 0
                total_recon_loss = 0

        if show_plots:
            plot_examples(obs, recon_0_steps, recon_t_steps)


def plot_examples(obs, recon_0_steps, recon_t_steps):
    o_0, o_1 = obs[0, 0].cpu().detach().numpy(), obs[0, 1].cpu().detach().numpy()
    recon_direct = recon_0_steps[0, 0].cpu().detach().numpy()
    recon_indirect = recon_t_steps[0, 0].cpu().detach().numpy()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(o_0)
    axs[0, 0].set_title("o_0")
    axs[0, 1].imshow(recon_direct)
    axs[0, 1].set_title("decoder(encoder(o_0))")
    axs[1, 0].imshow(o_1)
    axs[1, 0].set_title("o_1")
    axs[1, 1].imshow(recon_indirect)
    axs[1, 1].set_title("decoder(encoder(o_0) * phi(g_0))")
    # make sure that the plots are not cut off
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
