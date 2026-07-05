import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A sequence of raised balance beams with alternating lateral offsets and short gaps."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe zone
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Make the course a repeated "balance beam over pit" skill
    # Beam width stays challenging but realistic for a quadruped.
    beam_width_m = 0.55 + 0.15 * (1.0 - difficulty)   # wider at low difficulty
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(0.4))

    # Beam height and gap length scale with difficulty
    beam_height = 0.03 + 0.14 * difficulty
    gap_len_m = 0.28 + 0.55 * difficulty
    gap_len = max(m_to_idx(gap_len_m), m_to_idx(0.4))

    # Beam segment length
    beam_len_m = 0.85 + 0.15 * (1.0 - difficulty)
    beam_len = max(m_to_idx(beam_len_m), m_to_idx(0.4))

    # Lateral offsets alternate left/right to force small turns between goals
    max_offset_m = 0.55
    offset_steps = [-0.45, 0.35, -0.20, 0.50, -0.35, 0.25, -0.55]
    offset_scale = 0.35 + 0.65 * difficulty

    # Place a pit after the spawn so the robot must get onto the first beam
    height_field[spawn_x:, :] = -0.9

    cur_x = spawn_x

    for i in range(7):
        # Alternate lateral position with difficulty-dependent amplitude
        y_offset = int(round(offset_steps[i] * offset_scale * m_to_idx(max_offset_m)))
        center_y = int(np.clip(mid_y + y_offset, beam_width // 2 + 1, W - beam_width // 2 - 2))

        # Ensure the beam fits inside the terrain bounds
        x1 = cur_x
        x2 = min(cur_x + beam_len, L - 1)
        y1 = max(center_y - beam_width // 2, 0)
        y2 = min(center_y + beam_width // 2 + 1, W)

        # Create the raised beam
        height_field[x1:x2, y1:y2] = beam_height

        # Put a goal near the middle of each beam
        goals[i] = [x1 + max((x2 - x1) // 2, 1), center_y]

        # Carve the next gap
        cur_x = x2 + gap_len
        if cur_x >= L:
            cur_x = L - 1
            break
        height_field[x2:cur_x, :] = -0.9

    # Final goal on the last beam/landing area
    final_x = min(cur_x, L - 2)
    final_y = int(np.clip(mid_y + int(round(offset_steps[6] * offset_scale * m_to_idx(max_offset_m))), 0, W - 1))
    goals[7] = [final_x, final_y]

    # If the final beam didn't reach the end, keep the remainder as flat ground after the last challenge
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Ensure the spawn region remains flat
    height_field[:spawn_x, :] = 0.0

    return height_field, goals