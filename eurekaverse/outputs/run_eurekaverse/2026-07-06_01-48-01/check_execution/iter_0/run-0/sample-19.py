import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight corridor of alternating balance beams and low pit crossings to test precise foot placement and jump timing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic dimensions
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep spawn area clear
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Course parameters: repeated narrow beams over pits
    # Difficulty increases beam narrowness, pit depth, and spacing
    beam_width_m = 0.70 - 0.20 * difficulty   # around robot body width, but still walkable
    beam_width = max(m_to_idx(0.40), m_to_idx(beam_width_m))
    beam_thickness = m_to_idx(0.18 - 0.05 * difficulty)  # longitudinal landing area
    beam_height = 0.08 + 0.18 * difficulty

    pit_depth = -(0.35 + 0.55 * difficulty)
    gap_len = m_to_idx(0.55 + 0.25 * difficulty)
    beam_gap = m_to_idx(0.45 + 0.15 * difficulty)

    # Small lateral offsets encourage precise alignment but keep straight-line goal progression
    lateral_choices = [
        0,
        m_to_idx(0.12),
        -m_to_idx(0.12),
        m_to_idx(0.20),
        -m_to_idx(0.20),
    ]

    def add_beam(x1, x2, y_center, h):
        """Add a raised rectangular beam."""
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = h

    def add_pit(x1, x2):
        """Add a flat pit section."""
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1:
            height_field[x1:x2, :] = pit_depth

    # Build eight goal locations: spawn, then alternating beam centers and recovery points
    cur_x = spawn_x
    goals[0] = [spawn_x - m_to_idx(0.4), mid_y]

    # Start with a short flat runup before the first beam
    runup_end = min(L, cur_x + m_to_idx(0.6))
    height_field[cur_x:runup_end, :] = 0.0
    cur_x = runup_end

    for i in range(1, 8):
        if i % 2 == 1:
            # Raised beam segment
            y_off = lateral_choices[(i + int(difficulty * 10)) % len(lateral_choices)]
            y_c = int(np.clip(mid_y + y_off, beam_width // 2, W - beam_width // 2 - 1))

            x1 = cur_x
            x2 = min(L, cur_x + beam_thickness)
            add_beam(x1, x2, y_c, beam_height)

            # Goal centered on the beam
            goals[i] = [x1 + (x2 - x1) / 2.0, y_c]

            # Advance past beam and insert a pit
            cur_x = x2
            pit_end = min(L, cur_x + gap_len)
            add_pit(cur_x, pit_end)
            cur_x = pit_end

        else:
            # Recovery flat landing strip after the pit
            landing_len = m_to_idx(0.8 + 0.15 * difficulty)
            x1 = cur_x
            x2 = min(L, cur_x + landing_len)
            height_field[x1:x2, :] = 0.0

            goals[i] = [x1 + (x2 - x1) / 2.0, mid_y]

            cur_x = x2
            # Add another pit to keep the robot jumping between usable surfaces
            pit_end = min(L, cur_x + beam_gap)
            add_pit(cur_x, pit_end)
            cur_x = pit_end

    # Ensure the final region is usable and the course ends flat
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Clamp goals to valid bounds and ensure integer indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals