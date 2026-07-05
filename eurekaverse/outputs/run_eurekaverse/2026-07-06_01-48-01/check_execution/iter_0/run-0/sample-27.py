import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A line of broad stepping-stones over shallow pits, testing repeated jump precision and landing control."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe flat zone
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Use a pit after the spawn so the robot must stay on the stones
    pit_height = -0.8 - 0.6 * difficulty
    height_field[spawn_len:, :] = pit_height

    # Stepping-stone course parameters
    # Stones get slightly longer and gaps get slightly smaller at lower difficulty
    stone_len_m = 0.65 + 0.15 * (1.0 - difficulty)
    stone_gap_m = 0.42 + 0.55 * difficulty
    stone_w_m = 1.10 + 0.20 * random.random()  # wide enough for the robot, with mild variation
    stone_h_min = 0.10 + 0.05 * difficulty
    stone_h_max = 0.22 + 0.12 * difficulty

    stone_len = max(m_to_idx(0.4), m_to_idx(stone_len_m))
    stone_gap = max(m_to_idx(0.4), m_to_idx(stone_gap_m))
    stone_w = max(m_to_idx(1.0), m_to_idx(stone_w_m))

    # Keep stones centered but allow a small lateral wiggle to require minor steering
    y_jitter_max = m_to_idx(0.18 + 0.10 * difficulty)
    x = spawn_len + m_to_idx(0.40)

    # First goal: just before the first stone
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    def add_stone(x1, x2, y_center, h):
        """Place a rectangular stepping stone."""
        half_w = stone_w // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        height_field[x1:x2, y1:y2] = h

    # Create 6 stepping stones; the 7th and 8th goals lead off the final stone to flat ground
    stone_centers = []
    for i in range(6):
        if x >= L - m_to_idx(1.0):
            break

        y_offset = random.randint(-y_jitter_max, y_jitter_max) if y_jitter_max > 0 else 0
        y_center = int(np.clip(mid_y + y_offset, stone_w // 2, W - stone_w // 2 - 1))
        h = random.uniform(stone_h_min, stone_h_max)

        x2 = min(L, x + stone_len)
        add_stone(x, x2, y_center, h)
        stone_centers.append((x, x2, y_center))

        # Goal near the middle of each stone
        goals[i + 1] = [(x + x2) / 2.0, y_center]

        # Create the gap after the stone
        x = x2 + stone_gap

    # Final flat landing zone after the last stone
    landing_start = min(L, x)
    height_field[landing_start:, :] = 0.0

    # If we placed fewer than 6 stones due to terrain length, pad the remaining goals at the landing zone
    if len(stone_centers) == 0:
        stone_centers = [(spawn_len, spawn_len + stone_len, mid_y)]

    last_stone = stone_centers[-1]
    last_x2 = last_stone[1]
    last_y = last_stone[2]

    # Goals 6 and 7: from the final stone toward the landing area
    goals[6] = [min(L - 1, last_x2 - m_to_idx(0.15)), last_y]
    goals[7] = [min(L - 1, landing_start + m_to_idx(0.8)), mid_y]

    # Ensure all goals are within bounds
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals