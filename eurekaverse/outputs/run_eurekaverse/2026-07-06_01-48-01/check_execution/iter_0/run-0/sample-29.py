import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight sequence of low raised balance beams over pits with slight lateral offsets."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2), dtype=np.int16)

    # Useful constants
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2
    spawn_x = m_to_idx(2.0)  # keep first 2 m flat for spawning
    x_end = L

    # Skill target: controlled foot placement and balance on narrow elevated beams
    # The robot must repeatedly step onto a narrow beam, traverse it, then step back down.
    # We use pits on both sides to discourage bypassing and force use of the beam.

    # Beam parameters, scaled by difficulty
    beam_height = 0.03 + 0.08 * difficulty  # low enough to be realistic but still require balance
    beam_half_width = m_to_idx(0.22 + 0.03 * (1.0 - difficulty))  # total width ~0.44-0.50 m (narrow but allowed)
    beam_length = m_to_idx(1.25 + 0.35 * difficulty)

    # Gap and approach lengths
    gap_length = m_to_idx(0.55 + 0.25 * difficulty)
    segment_length = beam_length + gap_length

    # Lateral offsets for each beam, still keeping the course centered overall
    # Small shifts demand steering while preserving forward progression.
    y_offsets_m = [
        0.00,
        0.18 - 0.10 * difficulty,
        -0.20 + 0.08 * difficulty,
        0.16,
        -0.14,
        0.10 - 0.05 * difficulty,
        -0.08,
        0.00
    ]
    y_offsets = [int(np.clip(m_to_idx(o), -W // 4, W // 4)) for o in y_offsets_m]

    # Make the course mostly a pit so the robot must stay on the elevated beams.
    # Keep spawn region flat.
    height_field[spawn_x:, :] = -0.9
    height_field[:spawn_x, :] = 0.0

    def add_beam(x1, x2, cy, half_w, h):
        """Add a raised rectangular beam."""
        y1 = max(0, cy - half_w)
        y2 = min(W, cy + half_w + 1)
        x1 = max(0, x1)
        x2 = min(L, x2)
        height_field[x1:x2, y1:y2] = h

    def add_entry_exit_ramps(x1, x2, cy, half_w, h):
        """Add short ramps leading onto and off the beam for smoother transitions."""
        ramp_len = max(1, m_to_idx(0.25))
        # Entry ramp
        for i in range(ramp_len):
            frac = (i + 1) / ramp_len
            xi1 = max(0, x1 - ramp_len + i)
            xi2 = max(0, x1 - ramp_len + i + 1)
            y1 = max(0, cy - half_w)
            y2 = min(W, cy + half_w + 1)
            height_field[xi1:xi2, y1:y2] = h * frac
        # Exit ramp
        for i in range(ramp_len):
            frac = 1.0 - (i + 1) / ramp_len
            xi1 = min(L, x2 + i)
            xi2 = min(L, x2 + i + 1)
            y1 = max(0, cy - half_w)
            y2 = min(W, cy + half_w + 1)
            height_field[xi1:xi2, y1:y2] = max(height_field[xi1:xi2, y1:y2].max(), h * frac)

    # Place the first goal near the end of the spawn area
    goals[0] = [spawn_x - m_to_idx(0.35), mid_y]

    cur_x = spawn_x + m_to_idx(0.2)

    for i in range(7):
        cy = int(np.clip(mid_y + y_offsets[i], beam_half_width + 1, W - beam_half_width - 2))

        # Add pit before the beam to prevent bypassing without stepping up
        pit_start = cur_x
        pit_end = min(L, cur_x + gap_length)
        height_field[pit_start:pit_end, :] = -0.9

        # Add the beam
        beam_start = pit_end
        beam_end = min(L, beam_start + beam_length)
        add_beam(beam_start, beam_end, cy, beam_half_width, beam_height)
        add_entry_exit_ramps(beam_start, beam_end, cy, beam_half_width, beam_height)

        # Place goal near the center of each beam
        goal_x = beam_start + (beam_end - beam_start) // 2
        goals[i + 1] = [goal_x, cy]

        # Prepare for next obstacle
        cur_x = beam_end

    # Ensure the last section is still a pit, but provide a flat landing zone near the end
    landing_start = max(cur_x, L - m_to_idx(1.0))
    height_field[landing_start:, :] = 0.0

    # Final goal on the landing zone
    goals[7] = [min(L - 1, landing_start + m_to_idx(0.5)), mid_y]

    # Keep goal indices within bounds
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals