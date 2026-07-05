import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A line of staggered raised balance beams with short gaps, testing precise stepping and lateral balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain dimensions
    L = m_to_idx(length)
    W = m_to_idx(width)
    height_field = np.zeros((L, W), dtype=np.float32)
    goals = np.zeros((8, 2), dtype=np.float32)

    # Fixed corridor centered in y
    mid_y = W // 2

    # Keep spawn area flat and safe
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # The robot should traverse a consistent sequence of narrow elevated beams.
    # We use a pit below the beams so the robot must stay on top and cannot simply step off and walk around.
    pit_height = -0.9 - 0.4 * difficulty
    height_field[spawn_len:, :] = pit_height

    # Beam / platform geometry
    beam_len_m = 0.85 - 0.15 * difficulty
    beam_len = max(m_to_idx(beam_len_m), m_to_idx(0.4))

    beam_w_m = 1.1 - 0.1 * difficulty
    beam_w = max(m_to_idx(beam_w_m), m_to_idx(1.0))

    gap_m = 0.20 + 0.55 * difficulty
    gap = max(m_to_idx(gap_m), m_to_idx(0.15))

    # Beam height: modest but noticeable; harder courses are slightly higher
    beam_h = 0.08 + 0.18 * difficulty

    # Small lateral shifts to force side-to-side balancing while remaining realistic
    y_shift_choices = [-0.45, -0.25, 0.0, 0.25, 0.45]
    y_shift_scale = int(round((1 + difficulty * 1.5)))

    # Place 8 goals, one per beam/transition point
    cur_x = spawn_len
    cur_y = mid_y

    def clamp_y(center_y, half_w):
        return max(half_w, min(W - half_w - 1, center_y))

    for i in range(8):
        # Vary beam center slightly as difficulty increases
        if i > 0:
            shift_m = random.choice(y_shift_choices) * difficulty
            cur_y = int(round(cur_y + m_to_idx(shift_m)))
            cur_y = clamp_y(cur_y, beam_w // 2 + 1)

        x1 = cur_x
        x2 = min(cur_x + beam_len, L - 1)

        y1 = clamp_y(cur_y, beam_w // 2)
        y0 = y1 - beam_w // 2
        y2 = min(y0 + beam_w, W)

        # Create the beam/platform
        height_field[x1:x2, y0:y2] = beam_h

        # Goal near the center of each beam
        goals[i] = [x1 + (x2 - x1) * 0.55, y0 + (y2 - y0) / 2.0]

        # Advance to next beam with a gap in between
        cur_x = x2 + gap

        # Keep the remaining course beyond the last beam as pit, except the beam itself
        if cur_x >= L:
            break

    # If the final beam ended early, ensure the rest stays as pit
    if cur_x < L:
        height_field[cur_x:, :] = pit_height

    # Make sure spawn zone remains flat
    height_field[:spawn_len, :] = 0.0

    # Ensure all goals are valid indices within bounds
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals