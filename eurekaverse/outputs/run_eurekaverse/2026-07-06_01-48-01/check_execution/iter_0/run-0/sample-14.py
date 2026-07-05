import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight-line course of raised balance beams over recessed pits, testing precision stepping and body control."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain dimensions in indices
    L = height_field.shape[0]
    W = height_field.shape[1]
    mid_y = W // 2

    # --- Difficulty-scaled parameters ---
    # Beam gets narrower and gaps get longer as difficulty increases.
    beam_width_m = 1.25 - 0.45 * difficulty   # always at least 1.0m-ish, suitable for a quadruped
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(1.0))
    beam_height = 0.08 + 0.10 * difficulty    # raised enough to require stepping up, but still realistic

    # Segment structure: repeated beam + gap pattern
    start_x = m_to_idx(2.0)  # keep spawn area clear
    usable_end = L - m_to_idx(0.8)

    n_segments = 7  # 8 goals: spawn + 7 subsequent targets
    # Total course length is fixed; distribute segments across the arena
    total_available = usable_end - start_x
    base_seg = total_available // n_segments

    # Gap and beam proportions vary a bit with difficulty
    gap_m = 0.35 + 0.75 * difficulty
    gap = m_to_idx(gap_m)
    beam_len_m = max(0.9, 1.25 - 0.25 * difficulty)
    beam_len = m_to_idx(beam_len_m)

    # Random but bounded lateral offsets to force subtle side-stepping
    max_offset = m_to_idx(0.35 + 0.15 * difficulty)

    # Helper to safely place a rectangular raised beam
    def add_beam(x1, x2, y_center, height):
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = height

    # Spawn region: flat and safe
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0
    goals[0] = [spawn_end - m_to_idx(0.4), mid_y]

    # Build repeated beam-over-pit pattern
    cur_x = spawn_end
    cur_y = mid_y

    for i in range(1, 8):
        # Slight lateral drift to create mild steering without sharp turns
        if i < 7:
            cur_y = int(np.clip(cur_y + random.randint(-max_offset, max_offset), beam_width // 2, W - beam_width // 2 - 1))

        # Carve a pit in the upcoming region so the robot must stay on the beam
        pit_x1 = cur_x
        pit_x2 = min(L, cur_x + beam_len + gap)
        if pit_x2 > pit_x1:
            height_field[pit_x1:pit_x2, :] = -0.9 - 0.25 * difficulty

        # Place the beam across the pit
        beam_x1 = cur_x
        beam_x2 = min(L, cur_x + beam_len)
        add_beam(beam_x1, beam_x2, cur_y, beam_height)

        # Goal at the center of the beam
        goals[i - 0 if i == 0 else i - 1]  # no-op to keep indexing clear

        if i <= 7:
            # Place each intermediate goal near the beam center
            goals[i - 0] = [min(L - 1, beam_x1 + beam_len // 2), cur_y]

        # Advance to next segment
        cur_x += beam_len + gap

        # Stop if nearing the end; remaining goals are placed on the final safe platform
        if cur_x >= usable_end:
            break

    # Ensure the remaining course after the last beam is safe flat ground
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Final goal on a flat finish area
    finish_x = min(L - 2, max(cur_x + m_to_idx(0.6), L - m_to_idx(0.7)))
    goals[-1] = [finish_x, mid_y]

    # Make sure all goals are within bounds and integral indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals