import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """Stepping-stone log course with alternating narrow elevated beams over pits."""
    
    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -------------------------------------------------------------------------
    # Course idea:
    # A straight "balance and precision" course made of repeated narrow beams
    # spanning shallow pits. The robot must place its feet accurately on each
    # beam and transition cleanly across gaps without stepping into the pits.
    #
    # Skill tested:
    #   - precise foot placement
    #   - balance on narrow supports
    #   - controlled stepping over gaps
    #
    # Design notes:
    #   - Obstacles begin after the spawn safety region (x >= 2m).
    #   - Each beam is at least 1m long and narrow enough to challenge balance.
    #   - Pits are negative height to discourage shortcuts through the gaps.
    #   - The course is mostly straight and consistent, with repeated elements.
    # -------------------------------------------------------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn area: keep first 2 meters flat
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # Terrain parameters scaled by difficulty
    beam_length = m_to_idx(1.15 - 0.15 * difficulty)   # 1.15m -> 1.0m
    beam_width = m_to_idx(0.55 + 0.10 * difficulty)    # narrow, but stable enough
    beam_height = 0.08 + 0.18 * difficulty             # elevated but not too high

    gap_length = m_to_idx(0.55 + 0.55 * difficulty)    # larger gaps at higher difficulty
    pit_depth = -(0.55 + 0.35 * difficulty)

    # Small lateral drift to require gentle correction, but still mostly straight
    lateral_offsets_m = [0.0, 0.15, -0.10, 0.12, -0.14, 0.10, -0.08, 0.0]
    lateral_offsets = [m_to_idx(v) for v in lateral_offsets_m]

    def add_beam(x1, x2, center_y, h):
        """Place a rectangular elevated beam."""
        half_w = beam_width // 2
        y1 = max(0, center_y - half_w)
        y2 = min(W, center_y + half_w + 1)
        x1 = max(0, x1)
        x2 = min(L, x2)
        height_field[x1:x2, y1:y2] = h

    def add_pit(x1, x2):
        """Lower the ground to create a pit."""
        x1 = max(0, x1)
        x2 = min(L, x2)
        height_field[x1:x2, :] = pit_depth

    # Start with a short flat landing after spawn
    cur_x = spawn_end
    flat_start = m_to_idx(0.35)
    height_field[cur_x:cur_x + flat_start, :] = 0.0
    goals[0] = [cur_x + flat_start // 2, mid_y]

    cur_x += flat_start

    # Generate 6 repeating beam-over-pit segments
    for i in range(6):
        offset = lateral_offsets[i + 1]
        center_y = int(np.clip(mid_y + offset, beam_width // 2, W - beam_width // 2 - 1))

        # Add beam
        add_beam(cur_x, cur_x + beam_length, center_y, beam_height)
        goals[i + 1] = [cur_x + beam_length // 2, center_y]

        cur_x += beam_length

        # Add pit after the beam, except after the last beam we keep a flat finish
        if i < 5:
            add_pit(cur_x, cur_x + gap_length)
            cur_x += gap_length

    # Final flat runout to the end of the course
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Put the last goal near the final end of the course on flat ground
    final_goal_x = min(L - 1, cur_x + m_to_idx(0.6))
    goals[7] = [final_goal_x, mid_y]

    # Ensure all goals are within bounds and integer indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals