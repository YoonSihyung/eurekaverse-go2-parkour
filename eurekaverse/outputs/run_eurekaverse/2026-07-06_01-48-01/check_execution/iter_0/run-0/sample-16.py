import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight-line course of alternating raised balance rails and shallow gaps to test precision foot placement and jump timing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # ----------------------------
    # Course design:
    # - A flat spawn zone for safety
    # - A sequence of narrow raised rails separated by shallow pits
    # - The rails are wide enough to stand on, but narrow enough to force careful foot placement
    # - Difficulty increases by making the rails a bit longer/shorter, pits deeper, and lateral placement slightly less forgiving
    # - The robot moves straight along x, staying near the center line in y
    # ----------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep the first part flat so the robot does not spawn in an obstacle
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Course parameters, tuned to stay realistic for a quadruped
    # Rail width stays at least 1 m, as requested
    rail_width_m = 1.05 - 0.15 * difficulty
    rail_width = max(m_to_idx(rail_width_m), m_to_idx(1.0))

    # Rail height is modest so it can be stepped on, but still creates a precision task
    rail_height = 0.10 + 0.12 * difficulty

    # Pit depth makes the robot commit to staying on the rails
    pit_depth = -(0.15 + 0.25 * difficulty)

    # Each segment includes a rail and a gap
    rail_len_m = 0.85 - 0.15 * difficulty
    gap_len_m = 0.45 + 0.35 * difficulty
    rail_len = max(m_to_idx(rail_len_m), m_to_idx(0.4))
    gap_len = max(m_to_idx(gap_len_m), m_to_idx(0.4))

    # Small lateral variation to keep the course interesting without making it a turning course
    y_jitter_max = max(1, m_to_idx(0.12 + 0.10 * difficulty))

    # Start just after the spawn area
    cur_x = spawn_len + m_to_idx(0.4)

    # First goal near the start of the course
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    def place_rail(x1, x2, center_y):
        """Places a raised narrow rail and surrounds it with a pit."""
        half_w = rail_width // 2
        y1 = max(0, center_y - half_w)
        y2 = min(W, center_y + half_w + (rail_width % 2))
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = rail_height

    # Build 6 rail segments, with goals placed on each one
    for i in range(6):
        if cur_x >= L - m_to_idx(1.0):
            break

        # Slight random lateral offset, but keep the rails safely inside bounds
        dy = random.randint(-y_jitter_max, y_jitter_max)
        rail_center_y = int(np.clip(mid_y + dy, rail_width // 2 + 1, W - rail_width // 2 - 2))

        # Add pit before the rail
        pit_x1 = cur_x
        pit_x2 = min(L, cur_x + gap_len)
        height_field[pit_x1:pit_x2, :] = pit_depth

        # Place the rail itself
        rail_x1 = pit_x2
        rail_x2 = min(L, rail_x1 + rail_len)
        height_field[rail_x1:rail_x2, :] = pit_depth
        place_rail(rail_x1, rail_x2, rail_center_y)

        # Goal near the center of the rail
        gx = rail_x1 + (rail_x2 - rail_x1) // 2
        goals[i + 1] = [gx, rail_center_y]

        # Advance
        cur_x = rail_x2 + m_to_idx(0.35)

    # Fill remaining area after the last rail with flat ground so the final goal can be reached cleanly
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Final goal placed on safe flat ground after the last obstacle
    final_goal_x = min(L - 1, cur_x + m_to_idx(0.6))
    goals[-1] = [final_goal_x, mid_y]

    # Ensure spawn zone remains flat even if any earlier slice touched it
    height_field[:spawn_len, :] = 0.0

    return height_field, goals