from typing import Tuple, List

import torch


def hae_loss(
        observations: torch.Tensor, latent_observations: torch.Tensor,
        predicted_latents: torch.Tensor, predicted_observations: torch.Tensor,
        gamma: float,
        reconstruction_loss_fn=torch.nn.functional.mse_loss,
        prediction_loss_fn=torch.nn.functional.mse_loss
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert observations.shape == predicted_observations.shape, f"the shapes should match, but are {observations.shape} and {predicted_observations.shape}"
    assert latent_observations.shape == predicted_latents.shape, f"the shapes should match, but are {latent_observations.shape} and {predicted_latents.shape}"

    # as in the paper, the latent representation of the first observation in each trajectory is not considered
    pred_loss = prediction_loss_fn(latent_observations[:, 1:], predicted_latents[:, 1:])
    weighted_pred_loss = torch.mul(gamma, pred_loss)

    recon_loss = reconstruction_loss_fn(predicted_observations, observations)

    loss = torch.add(recon_loss, weighted_pred_loss)
    return loss, recon_loss, weighted_pred_loss


class HomomorphismAutoencoder(torch.nn.Module):
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    phi: torch.nn.Module
    encoder_input_shape: Tuple[int, ...]
    decoder_output_shape: Tuple[int, ...]
    latent_space_dim: int
    block_side_lengths: List[int]

    def __init__(
            self, encoder: torch.nn.Module, decoder: torch.nn.Module, phi_network: torch.nn.Module,
            encoder_input_shape: Tuple[int, ...], decoder_output_shape: Tuple[int, ...],
            latent_space_dimensionality: int, block_side_lengths: List[int]
    ):
        """

        :param encoder: this network has to output a vector of size latent_space_dimensionality
        :param decoder: this network has to take a vector of size latent_space_dimensionality as input
        :param phi_network: this network gets as input the vector parameterizing all the actions g_t.
        :param encoder_input_shape: used so that the HAE can take any shape of observations as input
        :param decoder_output_shape: used so that the HAE can output any shape of observations
        :param latent_space_dimensionality: size of the latent space
        :param block_side_lengths: the matrices in the latent space are block diagonal. This list contains the side
                lengths of the blocks. If sum(block_side_lengths) < latent_space_dimensionality, then the remaining
                values along the diagonal of the block diagonal matrix are set to 1.
                Note that the phi_network must produce enough values to fill the blocks.
        """
        super(HomomorphismAutoencoder, self).__init__()
        # TODO: make as many assertions as necessary
        self.encoder = encoder
        self.decoder = decoder
        self.phi = phi_network
        self.encoder_input_shape = encoder_input_shape
        self.decoder_output_shape = decoder_output_shape
        self.latent_space_dim = latent_space_dimensionality
        self.block_side_lengths = block_side_lengths

    def forward(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param observations: tensor of shape (batch_size, time_dimension, *encoder_input_shape)
        :param actions: tensor of shape (batch_size, time_dimension, action_dim)
        :return: The tensors output are:
                first tensor contains the latent representations of the observations
                    i.e. the h(o_{t+j}) from the paper
                    the tensor's shape is (batch_size, time_dimension, latent_space_dim)
                second tensor contains the block diagonal matrices
                    i.e. the rho(g_{t+i}) from the paper
                    with shape (batch_size, time_dimension, latent_space_dim, latent_space_dim)
                third tensor contains the predictions in latent space
                    i.e. the prod_{i=0}^{j-1} rho(g_{t+1}) h(o_t) from the paper
                    with shape (batch_size, time_dimension, latent_space_dimension)
                fourth tensor contains the predicted observations
                    i.e. the d(prod_{i=0}^{j-1} rho(g_{t+1}) h(o_t)) from the paper
                    with shape (batch_size, time_dimension, *decoder_output_shape)
        """
        assert observations.shape[1] - 1 == actions.shape[1], \
            f"one more observation than actions needed in a trajectory, " \
            f"but have {observations.shape[1]} and {actions.shape[1]}"

        # get the first output tensor
        latent_observations = self.latent_observations(observations)

        # the second output tensor
        rho = self.rho_of_g(actions)

        # the third output tensor
        batch_size, obs_time_dimension = latent_observations.shape[:2]
        predicted_latents = [latent_observations[:, 0].unsqueeze(1)]
        for i in range(1, obs_time_dimension):
            mat1 = predicted_latents[-1]
            mat2 = rho[:, i-1]
            predicted_latents.append(torch.bmm(mat1, mat2))
        predicted_latents = torch.stack(predicted_latents, dim=1).squeeze(2)

        # the fourth output tensor
        predicted_observations = self.reconstruct_observations(predicted_latents)

        return latent_observations, rho, predicted_latents, predicted_observations

    def latent_observations(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, time_dimension = observations.shape[:2]
        flattened_observations = observations.reshape(batch_size * time_dimension, *self.encoder_input_shape)
        flattened_latent_observations = self.encoder(flattened_observations)
        return flattened_latent_observations.reshape(batch_size, time_dimension, self.latent_space_dim)

    def reconstruct_observations(self, latent_observations: torch.Tensor) -> torch.Tensor:
        assert len(latent_observations.shape) == 3, f"latent_observations.shape is {latent_observations.shape}"
        assert latent_observations.shape[2] == self.latent_space_dim, \
            f"latent_observations.shape[2] is {latent_observations.shape[2]}"
        batch_size, time_dimension = latent_observations.shape[:2]
        flattened_latent_observations = latent_observations.reshape(batch_size * time_dimension, self.latent_space_dim)
        flattened_reconstructed_observations = self.decoder(flattened_latent_observations)
        return flattened_reconstructed_observations.reshape(batch_size, time_dimension, *self.decoder_output_shape)

    def rho_of_g(self, g: torch.Tensor) -> torch.Tensor:
        """
        computes the rho(g_t) from the paper
        :param g: tensor of shape (batch_size, time_dimension, phi_input_dimensionality)
        :return: tensor of shape (batch_size, time_dimension, latent_dimensionality, latent_dimensionality)
        """
        assert len(g.shape) == 3, f"g.shape is {g.shape}"
        batch_size, time_dimension = g.shape[:2]
        flattened_g = g.reshape(batch_size * time_dimension, -1)
        after_phi = self.phi(flattened_g)
        block_diagonal = self._create_block_diagonal_matrix(after_phi, batch_size, time_dimension)
        rho = torch.matrix_exp(block_diagonal)
        return rho.reshape(batch_size, time_dimension, self.latent_space_dim, self.latent_space_dim)

    def _create_block_diagonal_matrix(self, after_phi: torch.Tensor, batch_size: int, time_dimension: int):
        block_diagonal = torch.zeros(
            batch_size * time_dimension, self.latent_space_dim, self.latent_space_dim
        )
        block_diagonal = block_diagonal.to(after_phi.device)
        offset_diagonal = 0
        nbr_after_phi_values_used = 0
        for block_size in self.block_side_lengths:
            needed_after_phi_values = after_phi[:, nbr_after_phi_values_used:nbr_after_phi_values_used + block_size**2]
            block = needed_after_phi_values.reshape(batch_size * time_dimension, block_size, block_size)
            block = torch.transpose(block, 1, 2)
            block_diagonal[:,
                offset_diagonal:offset_diagonal + block_size,
                offset_diagonal:offset_diagonal + block_size
            ] = block
            offset_diagonal += block_size
            nbr_after_phi_values_used += block_size ** 2
        # note: through the matrix exponential, the values along the diagonal will automatically become ones
        return block_diagonal
