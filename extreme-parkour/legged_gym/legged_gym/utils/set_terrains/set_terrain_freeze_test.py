import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight-line sequence of raised balance beams separated by pits to train careful foot placement and jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Convenience values
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep the spawn area completely flat and safe
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Parameters for a repeated "beam over pit" skill.
    # The beam is wide enough for a quadruped but narrow enough to demand precision.
    beam_len_m = 0.9 - 0.15 * difficulty
    beam_len = max(m_to_idx(beam_len_m), m_to_idx(0.7))

    beam_width_m = 1.1 - 0.2 * difficulty
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(0.8))

    # Beam height increases slightly with difficulty to make the step-up more demanding.
    beam_height = 0.06 + 0.12 * difficulty

    # Pit depth creates a meaningful gap without making the course impossible.
    pit_depth = -(0.55 + 0.25 * difficulty)

    # Small horizontal gaps between beam segments
    gap_m = 0.45 + 0.25 * difficulty
    gap = max(m_to_idx(gap_m), m_to_idx(0.35))

    # Slightly vary centerline to make the robot maintain balance while staying mostly straight
    y_jitter = max(1, m_to_idx(0.12 + 0.18 * difficulty))

    def add_beam(x1, x2, y_center):
        """Adds a rectangular beam platform above the pit."""
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w + 1)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = beam_height

    # Create a pit after the spawn area so the robot must use the beams.
    height_field[spawn_len:, :] = pit_depth

    # Place the first goal near the end of the spawn region.
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    # Build 6 beam segments after spawn, then place final goal near the end.
    cur_x = spawn_len
    for i in range(6):
        # Slight alternating lateral offset to encourage subtle steering adjustments
        offset = ((-1) ** i) * random.randint(0, y_jitter)
        beam_y = int(np.clip(mid_y + offset, beam_width // 2, W - beam_width // 2 - 1))

        add_beam(cur_x, cur_x + beam_len, beam_y)

        # Put goal at the center of the beam segment
        goals[i + 1] = [cur_x + beam_len / 2, beam_y]

        # Advance over the gap
        cur_x += beam_len + gap

    # Make sure the remaining region is flat again for the finish
    finish_x = min(cur_x + m_to_idx(0.6), L - 1)
    height_field[cur_x:, :] = 0.0

    # Final goal on the flat landing zone
    goals[7] = [finish_x, mid_y]

    # Clip goals to valid grid bounds
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals