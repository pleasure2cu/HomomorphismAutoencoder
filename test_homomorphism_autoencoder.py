from typing import Tuple
from unittest import TestCase

import numpy as np
import torch

from homomorphism_autoencoder import HomomorphismAutoencoder, hae_loss


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

    def test_forward_1(self):
        hae = HomomorphismAutoencoder(
            _get_dummy_encoder_1(), _get_dummy_decoder_1(), _get_dummy_phi_1(), (15, 15), (15, 15), 10, [3, 1, 2]
        )
        observations = torch.ones((3, 3, 15, 15), dtype=torch.float)
        observations[1, 1, 6, 7] = 25.
        actions = torch.mul(4., _sample_after_phi_1())
        actions = actions.view(3, 2, 14)
        for i in range(3):
            actions[i, :, :] = torch.mul(1. / 2. ** (4 + i), actions[i, :, :])
        # latent_observations, rho, zero_step_recon, one_step_latent, one_step_recon = hae(observations, actions)
        latents, rho, predicted_latents, predicted_observations = hae(observations, actions)
        expected_latents = np.array([
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96, 9.96], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]],
            [[9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.], [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]]
        ], dtype=np.float32)
        self.assertTrue(torch.isclose(latents, torch.from_numpy(expected_latents)).all())
        self.assertEqual(rho.shape, (3, 2, 10, 10))
        self.assertEqual(predicted_latents.shape, (3, 3, 10))
        self.assertEqual(predicted_observations.shape, (3, 3, 15, 15))

        recon_layer = np.ones((15 * 15, 10), dtype=np.float32) / 25.
        for batch_index in range(3):
            ho1 = latents[batch_index][0].detach().numpy()
            recon = np.matmul(ho1, recon_layer.T).reshape((15, 15))
            self.assertTrue(np.isclose(recon, predicted_observations[batch_index][0].detach().numpy()).all())

            rho1 = rho[batch_index][0].detach().numpy()
            ho2_hat = np.matmul(ho1, rho1)
            recon = np.matmul(ho2_hat, recon_layer.T).reshape((15, 15))
            self.assertTrue(np.isclose(recon, predicted_observations[batch_index][1].detach().numpy()).all())

            rho2 = rho[batch_index][1].detach().numpy()
            ho3_hat = np.matmul(ho2_hat, rho2)
            recon = np.matmul(ho3_hat, recon_layer.T).reshape((15, 15))
            self.assertTrue(np.isclose(recon, predicted_observations[batch_index][2].detach().numpy()).all())


class TestLossFunctions(TestCase):
    def test_hae_loss_1(self):
        observations = np.random.random((4, 3, 3, 15, 20))
        latents = np.random.random((4, 3, 10))
        predicted_latents = np.random.random((4, 3, 10))
        predicted_observations = np.random.random((4, 3, 3, 15, 20))
        gamma = 300.
        loss, _, _ = hae_loss(
            torch.from_numpy(observations), torch.from_numpy(latents),
            torch.from_numpy(predicted_latents), torch.from_numpy(predicted_observations),
            gamma=gamma
        )

        pred_loss = np.mean((latents[:, 1:] - predicted_latents[:, 1:]) ** 2)
        weighted_pred_loss = gamma * pred_loss
        recon_loss = np.mean((observations - predicted_observations) ** 2)

        expected_loss = weighted_pred_loss + recon_loss

        self.assertTrue(torch.abs(loss - expected_loss) < 1e-6, msg=f"{loss} != {expected_loss}")
