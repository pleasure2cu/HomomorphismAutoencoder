from typing import Tuple

import numpy as np
import torch


def get_move_square_dataset(
        square_size: int, grid_size: int, nbr_steps: int, nbr_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    start_coordinates = np.random.randint(0, grid_size-square_size, size=(nbr_samples, 2))
    actions = np.random.randint(
        -int(grid_size/2+0.5), int(grid_size/2+0.5), size=(nbr_samples, nbr_steps, 2), dtype=np.int8
    )

    observations = np.zeros((nbr_samples, nbr_steps+1, grid_size, grid_size), dtype=np.float32)
    for i in range(nbr_samples):
        cur_pos = start_coordinates[i]
        observations[i, 0, cur_pos[0]:cur_pos[0]+square_size, cur_pos[1]:cur_pos[1]+square_size] = 1
        for j in range(nbr_steps):
            cur_pos += actions[i, j]
            cur_pos %= grid_size
            if cur_pos[0] > grid_size-square_size:
                cur_pos[0] += square_size
                cur_pos[0] %= grid_size
            if cur_pos[1] > grid_size-square_size:
                cur_pos[1] += square_size
                cur_pos[1] %= grid_size
            assert 0 <= cur_pos[0] <= grid_size-square_size
            assert 0 <= cur_pos[1] <= grid_size-square_size
            observations[i, j+1, cur_pos[0]:cur_pos[0]+square_size, cur_pos[1]:cur_pos[1]+square_size] = 1

    return observations, actions.astype(np.float32)


def get_u_rotation_dataset(
        u_side_length: int, grid_side_length: int, nbr_steps: int, nbr_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert u_side_length > 0 and u_side_length % 3 == 0

    block_size = u_side_length // 3
    u_0_deg = np.zeros((u_side_length, u_side_length), dtype=np.float32)
    u_0_deg[:, :block_size] = 1
    u_0_deg[:, 2*block_size:] = 1
    u_0_deg[-block_size:, :] = 1

    offset = (grid_side_length - u_side_length) // 2
    template_0_deg = np.zeros((grid_side_length, grid_side_length), dtype=np.float32)
    template_0_deg[offset:offset+u_side_length, offset:offset+u_side_length] = u_0_deg

    start_rots = np.random.randint(0, 4, size=nbr_samples, dtype=np.int8)
    actions = np.random.randint(0, 4, size=(nbr_samples, nbr_steps, 1), dtype=np.int8)
    observations = np.zeros((nbr_samples, nbr_steps+1, grid_side_length, grid_side_length), dtype=np.float32)
    for i in range(nbr_samples):
        base_of_sample = template_0_deg.copy()
        observations[i, 0] = np.rot90(base_of_sample, start_rots[i])
        total_rot = start_rots[i]
        for j, delta_rot in enumerate(actions[i]):
            total_rot += delta_rot[0]
            total_rot %= 4
            observations[i, j+1] = np.rot90(base_of_sample, total_rot)

    return observations, actions.astype(np.float32)


def get_move_rotation_u_dataset(
        u_side_length: int, grid_side_length: int, nbr_steps: int, nbr_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert u_side_length > 0 and u_side_length % 3 == 0
    assert False, "Not implemented yet"
    # todo: complete implementation

    block_size = u_side_length // 3
    u_0_deg = np.zeros((u_side_length, u_side_length), dtype=np.float32)
    u_0_deg[:, :block_size] = 1
    u_0_deg[:, 2*block_size:] = 1
    u_0_deg[-block_size:, :] = 1

    start_rots = np.random.randint(0, 4, size=nbr_samples, dtype=np.int8)
    start_trans = np.random.randint(0, grid_side_length-u_side_length, size=(nbr_samples, 2), dtype=np.int8)

    actions_rot = np.random.randint(0, 4, size=(nbr_samples, nbr_steps, 1), dtype=np.int8)
    actions_trans = np.random.randint(-int(grid_side_length/2+0.5), int(grid_side_length/2+0.5), size=(nbr_samples, nbr_steps, 2), dtype=np.int8)
    actions = np.concatenate((actions_trans, actions_rot), axis=-1)
    observations = np.zeros((nbr_samples, nbr_steps+1, grid_side_length, grid_side_length), dtype=np.float32)
    for i in range(nbr_samples):
        u_for_sample = u_0_deg.copy()
        u_for_sample = np.rot90(u_for_sample, start_rots[i])
        dim0, dim1 = start_trans[i]
        observations[i, 0, dim0: dim0+u_side_length, dim1: dim1+u_side_length] = u_for_sample

    return observations, actions.astype(np.float32)


if __name__ == "__main__":
    observations, actions = get_move_square_dataset(2, 8, 1, 30)
    #observations, actions = get_u_rotation_dataset(6, 8, 2, 30)
    observations = torch.from_numpy(observations)
    actions = torch.from_numpy(actions)

    dataset = torch.utils.data.TensorDataset(observations, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (obs, acts) in enumerate(dataloader):
        print(obs.shape, acts.shape)
        print(obs[0])
        print(acts[0])
        break
