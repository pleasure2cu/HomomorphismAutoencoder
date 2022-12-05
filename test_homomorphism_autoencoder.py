from typing import Tuple
from unittest import TestCase

import numpy as np
import torch

from homomorphism_autoencoder import HomomorphismAutoencoder, latent_prediction_loss, reconstruction_loss, hae_loss


def _sample_after_phi_1() -> torch.Tensor:
    return torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    ])


def _expected_block_diagonal_1() -> torch.Tensor:
    base_expected_block_diagonal_1 = torch.tensor([
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        [4, 5, 6, 0, 0, 0, 0, 0, 0, 0],
        [7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 11, 12, 0, 0, 0, 0],
        [0, 0, 0, 0, 13, 14, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float).T
    base_expected_block_diagonal_2 = torch.tensor([
        [2, 4, 6, 0, 0, 0, 0, 0, 0, 0],
        [8, 10, 12, 0, 0, 0, 0, 0, 0, 0],
        [14, 16, 18, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 20, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 22, 24, 0, 0, 0, 0],
        [0, 0, 0, 0, 26, 28, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float).T
    base_expected_block_diagonal_3 = torch.tensor([
        [3, 6, 9, 0, 0, 0, 0, 0, 0, 0],
        [12, 15, 18, 0, 0, 0, 0, 0, 0, 0],
        [21, 24, 27, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 30, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 33, 36, 0, 0, 0, 0],
        [0, 0, 0, 0, 39, 42, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float).T
    expected_block_diagonal = torch.stack([
        base_expected_block_diagonal_1,
        base_expected_block_diagonal_1,
        base_expected_block_diagonal_2,
        base_expected_block_diagonal_2,
        base_expected_block_diagonal_3,
        base_expected_block_diagonal_3,
    ])
    assert expected_block_diagonal.shape == (6, 10, 10)
    return expected_block_diagonal


class DummyNetwork(torch.nn.Module):
    def __init__(self, shape: Tuple[int, int], output_shape: Tuple):
        super(DummyNetwork, self).__init__()
        self.output_shape = output_shape
        self.linear = torch.nn.Linear(*shape)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, *self.output_shape)
        return x

    @staticmethod
    def initialize_weights(w, matrix):
        if isinstance(w, torch.nn.Linear):
            w.weight.data = matrix
            w.bias.data = torch.zeros(w.bias.data.shape)


def _get_dummy_encoder_1():
    matrix = torch.ones((10, 15 * 15), dtype=torch.float) / 25.
    encoder = DummyNetwork((15 * 15, 10), (10,))
    encoder.apply(lambda w: DummyNetwork.initialize_weights(w, matrix))
    return encoder


def _get_dummy_decoder_1():
    matrix = torch.ones((15 * 15, 10), dtype=torch.float) / 25.
    decoder = DummyNetwork((10, 15 * 15), (15, 15))
    decoder.apply(lambda w: DummyNetwork.initialize_weights(w, matrix))
    return decoder


def _get_dummy_phi_1():
    matrix = torch.eye(14, dtype=torch.float) / 4.
    phi = DummyNetwork((14, 14), (14,))
    phi.apply(lambda w: DummyNetwork.initialize_weights(w, matrix))
    return phi


class TestHomomorphismAutoencoder(TestCase):
    def test_create_block_diagonal_matrix_1(self):
        hae = HomomorphismAutoencoder(None, None, None, None, None, 10, [3, 1, 2])
        after_phi = _sample_after_phi_1()
        batch_size = 2
        time_dimension = 3
        block_diagonal = hae._create_block_diagonal_matrix(after_phi, batch_size, time_dimension)
        expected_block_diagonal = _expected_block_diagonal_1()
        self.assertTrue(torch.isclose(block_diagonal, expected_block_diagonal).all())

    def test_rho_of_g_1(self):
        hae = HomomorphismAutoencoder(None, None, _get_dummy_phi_1(), None, None, 10, [3, 1, 2])
        g = torch.mul(4., _sample_after_phi_1())
        g = g.view(3, 2, 14)
        rho_of_g = hae.rho_of_g(g)
        expected_rho_of_g = torch.matrix_exp(_expected_block_diagonal_1().view(3, 2, 10, 10))
        self.assertTrue(torch.isclose(rho_of_g, expected_rho_of_g).all())

    def test_latent_observations_1(self):
        hae = HomomorphismAutoencoder(_get_dummy_encoder_1(), None, None, (15, 15), None, 10, None)
        x = torch.ones((2, 3, 15, 15), dtype=torch.float)
        x[1, 1, 6, 7] = 25.
        latent_observations = hae.latent_observations(x)
        expected_latent_observations = np.ones((2, 3, 10), dtype=np.float32) / 25. * 15. * 15.
        expected_latent_observations[1, 1, :] += 1.0 - 1. / 25.
        self.assertTrue(torch.isclose(latent_observations, torch.from_numpy(expected_latent_observations)).all())

    def test_reconstruct_observations(self):
        hae = HomomorphismAutoencoder(None, _get_dummy_decoder_1(), None, None, (15, 15), 10, None)
        latent_observations = torch.arange(10, dtype=torch.float).view(1, 1, 10)
        latent_observations = latent_observations.repeat(1, 2, 1)
        latent_observations = latent_observations.repeat(3, 1, 1)
        assert latent_observations.shape == (3, 2, 10)
        reconstructed_observations = hae.reconstruct_observations(latent_observations)
        expected_reconstructed_observations = np.ones((3, 2, 15, 15), dtype=np.float32) / 25. * 45.
        self.assertTrue(
            torch.isclose(reconstructed_observations, torch.from_numpy(expected_reconstructed_observations)).all()
        )

    def test_forward_1_t_skip_default(self):
        hae = HomomorphismAutoencoder(_get_dummy_encoder_1(), _get_dummy_decoder_1(), _get_dummy_phi_1(), (15, 15),
                                      (15, 15), 10, [3, 1, 2])
        observations = torch.ones((3, 2, 15, 15), dtype=torch.float)
        observations[1, 1, 6, 7] = 25.
        actions = torch.mul(4., _sample_after_phi_1())
        actions = actions.view(3, 2, 14)
        for i in range(3):
            actions[i, :, :] = torch.mul(1. / 2. ** (4 + i), actions[i, :, :])
        latent_observations, rho, zero_step_recon, one_step_latent, one_step_recon = hae(observations, actions)
        expected_latent_observations = np.array([
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]]
        ], dtype=np.float32)
        expected_0_step_recon = np.ones((3, 2, 15, 15), dtype=np.float32) * 90. / 25.
        expected_0_step_recon[1, 1] = np.ones((15, 15), dtype=np.float32) * 99.6 / 25.
        self.assertTrue(torch.isclose(latent_observations, torch.from_numpy(expected_latent_observations)).all())
        self.assertTrue(torch.isclose(zero_step_recon, torch.from_numpy(expected_0_step_recon)).all())
        self.assertEqual(rho.shape, (3, 2, 10, 10))
        self.assertEqual(one_step_latent.shape, (3, 2, 10))
        self.assertEqual(one_step_recon.shape, (3, 2, 15, 15))

        recon_layer = np.ones((15 * 15, 10), dtype=np.float32) / 25.
        for batch_index in range(3):
            for time_index in range(2):
                rho_i_j = rho[batch_index, time_index].detach().numpy()
                lat_obs_i_j = latent_observations[batch_index, time_index].detach().numpy()
                latent_1_step = np.matmul(lat_obs_i_j, rho_i_j)
                expected_1_step_recon = np.matmul(latent_1_step, recon_layer.T).reshape((15, 15))
                self.assertTrue(
                    torch.isclose(
                        one_step_latent[batch_index, time_index],
                        torch.from_numpy(latent_1_step)
                    ).all(),
                    f"batch_index={batch_index}, time_index={time_index}"
                )
                self.assertTrue(
                    torch.isclose(
                        one_step_recon[batch_index, time_index],
                        torch.from_numpy(expected_1_step_recon)
                    ).all(),
                    f"batch_index: {batch_index}, time_index: {time_index}"
                )

    def test_forward_2_t_skip_2(self):
        hae = HomomorphismAutoencoder(_get_dummy_encoder_1(), _get_dummy_decoder_1(), _get_dummy_phi_1(), (15, 15),
                                      (15, 15), 10, [3, 1, 2])
        observations = torch.ones((3, 2, 15, 15), dtype=torch.float)
        observations[1, 1, 6, 7] = 25.
        actions = torch.mul(4., _sample_after_phi_1())
        actions = actions.view(3, 2, 14)
        for i in range(3):
            actions[i, :, :] = torch.mul(1. / 2. ** (4 + i), actions[i, :, :])
        latent_observations, rho, zero_step_recon, one_step_latent, one_step_recon = hae(observations, actions,
                                                                                         t_skip=2)
        expected_latent_observations = np.array([
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]]
        ], dtype=np.float32)
        expected_0_step_recon = np.ones((3, 2, 15, 15), dtype=np.float32) * 90. / 25.
        expected_0_step_recon[1, 1] = np.ones((15, 15), dtype=np.float32) * 99.6 / 25.
        self.assertTrue(torch.isclose(latent_observations, torch.from_numpy(expected_latent_observations)).all())
        self.assertTrue(torch.isclose(zero_step_recon, torch.from_numpy(expected_0_step_recon)).all())
        self.assertEqual(rho.shape, (3, 2, 10, 10))
        self.assertEqual(one_step_latent.shape, (3, 1, 10))
        self.assertEqual(one_step_recon.shape, (3, 1, 15, 15))

        recon_layer = np.ones((15 * 15, 10), dtype=np.float32) / 25.
        for batch_index in range(3):
            combined_rho = np.matmul(rho[batch_index, 0].detach().numpy(), rho[batch_index, 1].detach().numpy())
            lat_obs_i_j = latent_observations[batch_index, 0].detach().numpy()
            latent_1_step = np.matmul(lat_obs_i_j, combined_rho)
            expected_1_step_recon = np.matmul(latent_1_step, recon_layer.T).reshape((15, 15))
            self.assertTrue(
                torch.isclose(
                    one_step_latent[batch_index, 0],
                    torch.from_numpy(latent_1_step)
                ).all(),
                f"batch_index={batch_index}"
            )
            self.assertTrue(
                torch.isclose(
                    one_step_recon[batch_index, 0],
                    torch.from_numpy(expected_1_step_recon)
                ).all(),
                f"batch_index: {batch_index}, time_index: {0}"
            )


class TestLossFunctions(TestCase):
    def test_latent_prediction_loss_1(self):
        nbr_entries = 60
        true_latents = torch.arange(nbr_entries, dtype=torch.float).view(3, 2, 10)
        pred_latents = torch.arange(nbr_entries, dtype=torch.float)
        pred_latents[10] = 70.
        pred_latents[49] = 109.
        pred_latents = pred_latents.view(3, 2, 10)
        loss = latent_prediction_loss(pred_latents, true_latents, t_skip=0)
        self.assertEqual(loss, (60.**2 + 60**2) / nbr_entries)

    def test_latent_prediction_loss_2(self):
        true_latents = torch.arange(60, dtype=torch.float).view(3, 2, 10)
        pred_latents = torch.arange(60, dtype=torch.float)
        pred_latents[10] = 70.
        pred_latents[59] = 119.
        pred_latents = pred_latents.view(3, 2, 10)
        pred_latents = pred_latents[:, 1:, :]
        loss = latent_prediction_loss(pred_latents, true_latents, t_skip=1)
        self.assertEqual(loss, (60.**2 + 60**2) / 30)

    def test_reconstruction_loss_1(self):
        observations = np.random.random((3, 2, 15, 20, 3))
        reconstructions = np.random.random((3, 2, 15, 20, 3))
        loss = reconstruction_loss(torch.from_numpy(reconstructions), torch.from_numpy(observations), t_skip=0)
        expected_loss = np.mean((observations - reconstructions) ** 2)
        self.assertTrue(torch.abs(loss - expected_loss) < 1e-6)

    def test_reconstruction_loss_2(self):
        observations = np.random.random((3, 2, 15, 20, 3))
        reconstructions = np.random.random((3, 1, 15, 20, 3))
        loss = reconstruction_loss(torch.from_numpy(reconstructions), torch.from_numpy(observations), t_skip=1)
        expected_loss = np.mean((observations[:, 1:] - reconstructions) ** 2)
        self.assertTrue(torch.abs(loss - expected_loss) < 1e-6)

    def test_hae_loss_1(self):
        observations = np.random.random((4, 3, 15, 20, 3))
        zero_time_step_recon = np.random.random((4, 3, 15, 20, 3))
        t_time_step_recon = np.random.random((4, 1, 15, 20, 3))
        encoded_observations = np.random.random((4, 3, 10))
        t_time_step_latent = np.random.random((4, 1, 10))
        gamma = 300.
        loss, _, _ = hae_loss(
            torch.from_numpy(observations), torch.from_numpy(encoded_observations),
            torch.from_numpy(zero_time_step_recon), torch.from_numpy(t_time_step_recon),
            torch.from_numpy(t_time_step_latent), t_skip=2, gamma=gamma
        )

        pred_loss = np.mean((encoded_observations[:, 2:] - t_time_step_latent) ** 2)
        weighted_pred_loss = gamma * pred_loss
        full_true_obs = np.concatenate([observations, observations[:, 2:]], axis=1)
        full_recon_obs = np.concatenate([zero_time_step_recon, t_time_step_recon], axis=1)
        recon_loss = np.mean((full_true_obs - full_recon_obs) ** 2)

        expected_loss = weighted_pred_loss + recon_loss

        self.assertTrue(torch.abs(loss - expected_loss) < 1e-6), f"{loss} != {expected_loss}"
