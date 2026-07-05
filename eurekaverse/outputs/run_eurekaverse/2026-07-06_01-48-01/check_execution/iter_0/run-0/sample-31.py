import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A repeating raised balance beam and stepping-stone course that tests precision foot placement and straight-line jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -------------------------------------------------------------------------
    # Course design:
    # - A flat spawn zone for the first 2 meters.
    # - Then a repeated sequence of narrow raised beams separated by pits.
    # - Each beam is wide enough for a quadruped, but narrow enough to demand
    #   careful straight-line alignment and balance.
    # - Obstacles get slightly harder with difficulty by increasing height,
    #   reducing beam width, and widening the pits.
    # -------------------------------------------------------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep spawn area flat and obstacle-free
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Terrain parameters scaled by difficulty
    beam_length = m_to_idx(0.85 - 0.15 * difficulty)      # modest platform length
    beam_length = max(beam_length, m_to_idx(0.55))
    beam_width = m_to_idx(1.25 - 0.35 * difficulty)       # narrow but realistic
    beam_width = max(beam_width, m_to_idx(0.9))
    beam_height = 0.12 + 0.22 * difficulty                # clear step up/down
    pit_length = m_to_idx(0.35 + 0.45 * difficulty)      # gap between beams
    pit_length = max(pit_length, m_to_idx(0.25))

    # Lateral wobble of the beam centerline, but kept within bounds
    max_offset = m_to_idx(0.35)
    offset_choices = [-max_offset // 2, 0, max_offset // 2]

    def place_beam(x1, x2, y_center, height):
        """Places a rectangular raised beam centered at y_center."""
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = height

    # Set everything after spawn to a pit first, then place beams on top
    height_field[spawn_len:, :] = -0.55 - 0.15 * difficulty

    # Start just after spawn
    cur_x = spawn_len
    center_y = mid_y

    # First goal near the end of the spawn zone
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    # Build 7 traversable segments after the first goal, with 7 intermediate goals
    for i in range(7):
        # Slight center shifts to require alignment, but keep the path straight enough
        center_y = int(np.clip(center_y + random.choice(offset_choices), beam_width // 2, W - beam_width // 2 - 1))

        # Place beam segment
        place_beam(cur_x, cur_x + beam_length, center_y, beam_height)

        # Put goal near the middle of the beam segment
        goals[i + 1] = [cur_x + beam_length / 2, center_y]

        # Advance over the beam and the pit
        cur_x += beam_length + pit_length

        # If we reach the end, stop placing and keep remainder flat to allow completion
        if cur_x >= L - m_to_idx(1.0):
            break

    # If there is leftover terrain after the last obstacle, make it flat ground
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Ensure the final goal is on reachable flat terrain near the end
    goals[-1] = [min(L - m_to_idx(0.6), max(cur_x, spawn_len + m_to_idx(0.5))), center_y]

    # Clamp all goals to valid indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals