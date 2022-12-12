import numpy as np
import torch
import matplotlib.pyplot as plt

from networks import SmallDenseEncoder, SmallDenseDecoder, SmallDensePhi, SmallConvEncoder, SmallConvDecoder
from data_creation import get_move_square_dataset, get_u_rotation_dataset
from homomorphism_autoencoder import HomomorphismAutoencoder, hae_loss
from torchsummary import summary


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grid_size = 12
    latent_dim = 4
    nbr_samples = 100
    gamma = 0.0
    nbr_epochs = 2
    show_plots = True
    batch_size = 64
    # set up data
    observations, actions = get_move_square_dataset(square_size=3, grid_size=grid_size, nbr_steps=1, nbr_samples=nbr_samples)
    # observations, actions = get_u_rotation_dataset(u_side_length=6, grid_side_length=grid_size, nbr_steps=1, nbr_samples=nbr_samples)
    # observations = np.repeat(observations, 1000, axis=0)
    # actions = np.repeat(actions, 1000, axis=0)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # set up model
    # encoder = SmallDenseEncoder([grid_size ** 2, 128, 128, 128, 64, 32, 16, latent_dim])
    # decoder = SmallDenseDecoder([latent_dim, 16, 32, 64, 128, 128, 128, grid_size ** 2], (grid_size, grid_size))
    phi = SmallDensePhi([2, 128, 128, 128, 8])
    # hae = HomomorphismAutoencoder(encoder, decoder, phi, (grid_size, grid_size), (grid_size, grid_size), latent_dim, [2])
    encoder = SmallConvEncoder(
        nbr_channels_start=1, kernel_sizes=[3, 3, 3, 3], channels=[64, 64, 64, 64], strides=[1, 1, 1, 1],
        dense_widths=[4*4*64, 512, latent_dim]
    )
    summary(encoder, (1, grid_size, grid_size), device='cpu')
    decoder = SmallConvDecoder(
        kernel_sizes_deconv=[3, 3, 3, 3, 3], channels_deconv=[64, 64, 64, 64, 64], strides_deconv=[1, 1, 1, 1, 1],
        dense_widths=[latent_dim, 128, 4 * 4 * 64],
        kernel_sizes_conv=[3, 1], channels_conv=[32, 1], strides_conv=[1, 1],
        shape_after_dense=(64, 4, 4), remove_channel_at_end=True
    )
    summary(decoder, (latent_dim,), device='cpu')
    hae = HomomorphismAutoencoder(encoder, decoder, phi, (1, grid_size, grid_size), (grid_size, grid_size), latent_dim, [2, 2])
    hae.to(device)

    # get the optimizer
    optimizer = torch.optim.Adam(hae.parameters(), 0.001)

    # train model
    for epoch in range(nbr_epochs):
        print(f"Epoch {epoch + 1}/{nbr_epochs}")
        total_loss = 0
        total_pred_loss = 0
        total_recon_loss = 0
        for batch_idx, (obs, acts) in enumerate(dataloader):
            obs = obs.to(device)
            acts = acts.to(device)
            optimizer.zero_grad()

            latents, rho, predicted_latents, predicted_observations = hae(obs.unsqueeze(2), acts)
            loss, recon_loss, weighted_pred_loss = hae_loss(
                obs, latents, predicted_latents, predicted_observations, gamma=gamma
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
            plot_examples(obs, predicted_observations)


def plot_examples(obs, predicted_observations):
    o_0, o_1 = obs[0, 0].cpu().detach().numpy(), obs[0, 1].cpu().detach().numpy()
    recon_direct = predicted_observations[0, 0].cpu().detach().numpy()
    recon_indirect = predicted_observations[0, 1].cpu().detach().numpy()
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
