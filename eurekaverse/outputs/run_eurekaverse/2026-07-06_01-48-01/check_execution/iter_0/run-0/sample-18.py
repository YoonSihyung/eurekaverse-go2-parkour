import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A line of staggered balance beams and narrow gaps that tests precision stepping and controlled lateral placement."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -----------------------------
    # Course design:
    # - Start on flat ground.
    # - Then traverse a sequence of narrow raised beams separated by pits.
    # - Beam centers stay near the middle, but each beam shifts slightly left/right.
    # - The robot must place its feet carefully to stay on the beams.
    # - Difficulty increases beam narrowness, gap size, and lateral offset.
    # -----------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Safe spawn zone: keep first 2m flat and obstacle-free
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # Make the rest a pit so the robot must stay on the beams
    pit_depth = -0.65 - 0.15 * difficulty
    height_field[spawn_end:, :] = pit_depth

    # Beam parameters
    beam_length_m = 1.05 - 0.20 * difficulty
    beam_length = max(m_to_idx(beam_length_m), m_to_idx(0.45))

    # Narrow beams: allowed because the challenge is precise foot placement
    beam_width_m = 0.55 - 0.12 * difficulty
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(0.40))

    # Beam height, slightly above ground so stepping off is costly
    beam_height = 0.10 + 0.10 * difficulty

    # Gap between beams
    gap_m = 0.45 + 0.55 * difficulty
    gap = max(m_to_idx(gap_m), m_to_idx(0.35))

    # Lateral shift per beam, increasing with difficulty
    max_shift_m = 0.22 + 0.35 * difficulty
    max_shift = m_to_idx(max_shift_m)

    # Keep beams within bounds
    half_w = beam_width // 2
    x = spawn_end + m_to_idx(0.40)

    # First goal near the end of the spawn area
    goals[0] = [spawn_end - m_to_idx(0.25), mid_y]

    # Create 7 beam segments, with 8 goals total
    for i in range(7):
        # Shift beam left/right in a controlled way
        if i == 0:
            offset = 0
        else:
            # Alternate direction but keep it mostly centered
            direction = -1 if i % 2 == 0 else 1
            offset = int(direction * (0.35 * max_shift + random.randint(0, max(1, max_shift // 2))))

        center_y = int(np.clip(mid_y + offset, half_w + 1, W - half_w - 2))

        # Random but bounded beam length variation
        length_jitter = random.randint(-m_to_idx(0.10), m_to_idx(0.10))
        seg_len = int(np.clip(beam_length + length_jitter, m_to_idx(0.45), m_to_idx(1.25)))

        x1 = int(np.clip(x, spawn_end, L - 1))
        x2 = int(np.clip(x1 + seg_len, spawn_end, L))

        y1 = int(np.clip(center_y - half_w, 0, W))
        y2 = int(np.clip(center_y + half_w, 0, W))

        # Raise the beam above the pit
        height_field[x1:x2, y1:y2] = beam_height

        # Put a goal near the center of this beam
        goals[i + 1] = [x1 + (x2 - x1) / 2.0, center_y]

        # Advance to the next beam, leaving a gap of pit
        x = x2 + gap

        # Stop if we are near the end of the terrain
        if x >= L - m_to_idx(1.0):
            x = L - m_to_idx(1.0)

    # Final goal near the end of the last beam or on flat landing zone
    landing_start = min(max(int(x), spawn_end), L - 1)
    landing_end = L
    height_field[landing_start:landing_end, :] = pit_depth
    landing_x = max(landing_start + m_to_idx(0.6), L - m_to_idx(0.5))
    landing_y = mid_y
    goals[-1] = [landing_x, landing_y]

    # Ensure the final landing strip is flat ground
    landing_strip_start = max(L - m_to_idx(1.5), 0)
    height_field[landing_strip_start:, :] = 0.0

    return height_field, goals