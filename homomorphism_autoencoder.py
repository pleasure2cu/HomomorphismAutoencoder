from typing import Tuple, List

import torch


def latent_prediction_loss(
        predicted_latents: torch.Tensor,
        true_latents: torch.Tensor,
        t_skip: int = 1,
        loss_fn=torch.nn.functional.mse_loss
) -> torch.Tensor:
    assert len(true_latents.shape) == 3, f"true_latents must be of shape (batch_size, time_dim, latent_dim), got {true_latents.shape}"
    assert len(predicted_latents.shape) == 3, f"predicted_latents must be of shape (batch_size, time_dim, latent_dim), got {predicted_latents.shape}"
    assert true_latents.shape[0] == predicted_latents.shape[0], f"true_latents and predicted_latents must have the same batch_size, got {true_latents.shape[0]} and {predicted_latents.shape[0]}"
    assert true_latents.shape[1] == predicted_latents.shape[1] + t_skip, f"true_latents and predicted_latents must have the same time_dim up to t_skip, got {true_latents.shape[1]} and {predicted_latents.shape[1]} with t_skip={t_skip}"
    assert true_latents.shape[2] == predicted_latents.shape[2], f"true_latents and predicted_latents must have the same latent_dim, got {true_latents.shape[2]} and {predicted_latents.shape[2]}"

    loss = loss_fn(
        true_latents[:, t_skip:],
        predicted_latents
    )
    return loss


def reconstruction_loss(
        predicted_observations: torch.Tensor,
        true_observations: torch.Tensor,
        t_skip: int = 1,
        loss_fn=torch.nn.functional.mse_loss
) -> torch.Tensor:
    assert len(true_observations.shape) == len(predicted_observations.shape), f"true_observations and predicted_observations must have the same number of dimensions, got {true_observations.shape} and {predicted_observations.shape}"
    assert true_observations.shape[0] == predicted_observations.shape[0], f"true_observations and predicted_observations must have the same batch_size, got {true_observations.shape[0]} and {predicted_observations.shape[0]}"
    assert true_observations.shape[1] == predicted_observations.shape[1] + t_skip, f"true_observations and predicted_observations must have the same time_dim up to t_skip, got {true_observations.shape[1]} and {predicted_observations.shape[1]} with t_skip={t_skip}"
    assert true_observations.shape[2:] == predicted_observations.shape[2:], f"true_observations and predicted_observations must have the same spatial dimensions, got {true_observations.shape[2:]} and {predicted_observations.shape[2:]}"

    loss = loss_fn(
        true_observations[:, t_skip:],
        predicted_observations
    )
    return loss


def hae_loss(
        true_observations: torch.Tensor, encoded_observations: torch.Tensor,
        zero_timestep_reconstructions: torch.Tensor, t_timestep_reconstructions: torch.Tensor,
        t_timestep_latent_predictions: torch.Tensor, t_skip: int, gamma: float,
        base_loss_fn_rec=torch.nn.functional.mse_loss, base_loss_fn_pred=torch.nn.functional.mse_loss
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_pred_loss = latent_prediction_loss(
        t_timestep_latent_predictions, encoded_observations, t_skip, base_loss_fn_pred
    )
    zero_step_recon_loss = reconstruction_loss(
        zero_timestep_reconstructions, true_observations, t_skip=0, loss_fn=base_loss_fn_rec
    )
    t_step_recon_loss = reconstruction_loss(
        t_timestep_reconstructions, true_observations, t_skip=t_skip, loss_fn=base_loss_fn_rec
    )

    nbr_observations_per_trajectory = true_observations.shape[1]
    nbr_predictions_per_trajectory = t_timestep_reconstructions.shape[1]
    zero_step_weight = \
        nbr_observations_per_trajectory / (nbr_observations_per_trajectory + nbr_predictions_per_trajectory)
    t_step_weight = \
        nbr_predictions_per_trajectory / (nbr_observations_per_trajectory + nbr_predictions_per_trajectory)
    part1 = torch.mul(zero_step_weight, zero_step_recon_loss)
    part2 = torch.mul(t_step_weight, t_step_recon_loss)
    recon_loss = torch.add(part1, part2)

    weighted_pred_loss = torch.mul(gamma, latent_pred_loss)

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
                IMPORTANT: currently it is the responsibility of this network to keep the different parameters apart.
                    I.e. if for example phi takes the spatial shifts and the rotations as input, it has to make sure
                    that phi consists of two subnetworks - one for the spatial shifts and one for the rotations.
                    The output vector of phi is then the concatenation of the outputs of these two subnetworks.
                todo: rethink this
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
            self, observations: torch.Tensor, actions: torch.Tensor, t_skip: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param observations: tensor of shape (batch_size, time_dimension, *encoder_input_shape)
        :param actions: tensor of shape (batch_size, time_dimension, action_dim)
        :param t_skip: the number of actions that should be applied in latent space before a reconstruction
        :return: in the following, rho and g are used as in the HAE paper. I replaced 'h' and 'd' with 'encoder' and
        'decoder' for clarity. The tensors output are:
                first tensor is the latent representations of the observations with shape
                    (batch_size, time_dimension, latent_space_dim)
                second tensor is the rho(g_{t+i}), with shape
                    (batch_size, time_dimension, latent_space_dim, latent_space_dim)
                third tensor is the 0-step reconstructions of the observations, i.e. decoder(encoder(observations)),
                    with shape (batch_size, time_dimension, *decoder_output_shape)
                fourth tensor is the t_skip-step latents of the observations,
                    i.e. encoder(o_0) * rho(g_0) * ... * rho(g_{t_skip-1})
                    with shape (batch_size, time_dimension, latent_space_dim)
                fifth tensor is the t_skip-step reconstructions of the observations,
                    i.e. decoder(encoder(o_0) * rho(g_0) * ... * rho(g_{t_skip-1}))
                    with shape (batch_size, time_dimension, *decoder_output_shape)

        """
        # compute the basics (the latent representations of the observations and the block-diagonal matrices)
        latent_observations = self.latent_observations(observations)
        rho = self.rho_of_g(actions)

        # as a first easy step, compute the 0-step reconstructions
        zero_step_reconstructions = self.reconstruct_observations(latent_observations)

        # now compute the t_skip-step reconstructions
        assert t_skip > 0, f"t_skip is {t_skip}"
        batch_size, obs_time_dimension = latent_observations.shape[:2]
        actions_time_dimension = actions.shape[1]
        # todo: better assertion, because currently it is not possible to input 1 observation and multiple actions
        assert obs_time_dimension == actions_time_dimension+1 or obs_time_dimension == actions_time_dimension, \
            f"obs_time_dimension is {obs_time_dimension}, actions_time_dimension is {actions_time_dimension}"

        t_skip_step_latents = []
        for t in range(actions_time_dimension - t_skip + 1):
            h_o = latent_observations[:, t].unsqueeze(1)
            for i in range(t_skip):
                h_o = torch.bmm(h_o, rho[:, t+i])
            t_skip_step_latents.append(h_o)
        t_skip_step_latents = torch.cat(t_skip_step_latents, dim=1)
        t_skip_step_reconstructions = self.reconstruct_observations(t_skip_step_latents)
        return latent_observations, rho, zero_step_reconstructions, t_skip_step_latents, t_skip_step_reconstructions

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
