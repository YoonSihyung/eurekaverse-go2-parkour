import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone balance course with narrow elevated pads over a shallow trench."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # --- Basic dimensions and helpers ---
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2
    spawn_end = m_to_idx(2.0)  # keep first 2m flat for safe spawn
    step_h = 0.10 + 0.18 * difficulty  # elevated enough to require stepping, not climbing

    # Course style:
    # A line of narrow stepping stones/pads, each separated by a shallow trench.
    # The robot must repeatedly place its feet accurately and maintain balance while
    # traversing a consistent sequence of elevated surfaces.

    # Make the area after the spawn a trench so the robot cannot simply walk on flat ground
    # and must commit to the stepping stones.
    trench_depth = -0.35 - 0.25 * difficulty
    height_field[spawn_end:, :] = trench_depth

    # Keep the central corridor slightly wider than the robot body, but still narrow enough
    # to make precise placement necessary.
    corridor_half_width = m_to_idx(0.55 + 0.10 * difficulty) // 2
    corridor_half_width = max(corridor_half_width, m_to_idx(0.35))

    # Define stepping-stone geometry
    stone_len = m_to_idx(0.75 - 0.10 * difficulty)
    stone_len = max(stone_len, m_to_idx(0.45))
    stone_wid = m_to_idx(0.95 - 0.10 * difficulty)
    stone_wid = max(stone_wid, m_to_idx(0.40))

    gap_len = m_to_idx(0.45 + 0.35 * difficulty)
    gap_len = max(gap_len, m_to_idx(0.35))

    # Add a slight lateral offset pattern to force minor turning/weight shifts.
    y_offsets_m = [0.0, 0.18, -0.16, 0.20, -0.18, 0.15, -0.12, 0.0]
    y_offsets = [m_to_idx(v) for v in y_offsets_m]

    # Start building stones after the spawn zone
    cur_x = spawn_end + m_to_idx(0.25)

    for i in range(8):
        # Vary stone height a bit with difficulty to make the course more challenging
        # while keeping it realistic for a quadruped.
        h = step_h + random.uniform(-0.02, 0.03)

        # Compute stone center laterally with small offsets
        cy = int(np.clip(mid_y + y_offsets[i], stone_wid // 2 + 1, W - stone_wid // 2 - 2))

        x1 = int(cur_x)
        x2 = int(min(cur_x + stone_len, L))
        y1 = int(max(cy - stone_wid // 2, 0))
        y2 = int(min(cy + stone_wid // 2, W))

        # Place elevated stone
        height_field[x1:x2, y1:y2] = h

        # Put goal near the center of each stone
        goals[i, 0] = x1 + (x2 - x1) / 2
        goals[i, 1] = cy

        # Add a small flat "landing" strip on top of each stone to help stable traversal,
        # but keep the course challenging by surrounding it with trenches.
        landing_pad_len = max(m_to_idx(0.15), 1)
        lp1 = min(x2 - landing_pad_len, x1)
        lp2 = min(x1 + landing_pad_len, x2)
        height_field[lp1:lp2, y1:y2] = h

        # Advance to next stone
        cur_x = x2 + gap_len

        # Stop if we are running out of space; keep last goals inside bounds.
        if cur_x >= L - m_to_idx(0.8):
            break

    # If we produced fewer than 8 stones due to length constraints, extend the last stone region.
    last_goal_x = int(min(L - m_to_idx(0.5), max(goals[7, 0], spawn_end + m_to_idx(0.5))))
    for j in range(8):
        if goals[j, 0] == 0 and goals[j, 1] == 0:
            goals[j, 0] = last_goal_x
            goals[j, 1] = mid_y

    # Ensure spawn area remains perfectly flat.
    height_field[:spawn_end, :] = 0.0

    # Ensure all goals are within bounds and integer-like grid indices.
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals