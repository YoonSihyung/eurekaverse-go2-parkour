import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight corridor of alternating narrow beams and raised stepping pads over pits, testing balance and controlled jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain size: exactly 12m x 4m at 5cm resolution
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    x_max = height_field.shape[0]
    y_max = height_field.shape[1]
    mid_y = y_max // 2

    # Convert difficulty into useful parameters
    # Higher difficulty -> narrower beams, larger lateral offsets, longer pits
    beam_height = 0.10 + 0.08 * difficulty
    pad_height = 0.06 + 0.10 * difficulty
    pit_depth = -0.65 - 0.25 * difficulty

    beam_len_m = 1.0 - 0.15 * difficulty
    beam_w_m = 1.20 - 0.45 * difficulty
    pad_len_m = 0.70 - 0.10 * difficulty
    pad_w_m = 1.10 - 0.20 * difficulty
    gap_m = 0.35 + 0.35 * difficulty

    beam_len = max(8, m_to_idx(beam_len_m))
    beam_w = max(8, m_to_idx(beam_w_m))
    pad_len = max(6, m_to_idx(pad_len_m))
    pad_w = max(8, m_to_idx(pad_w_m))
    gap = max(4, m_to_idx(gap_m))

    # Spawn zone must remain flat and obstacle-free
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Helper to safely write a rectangular obstacle/pad
    def add_rect(x1, x2, y1, y2, h):
        x1 = max(0, min(x_max, x1))
        x2 = max(0, min(x_max, x2))
        y1 = max(0, min(y_max, y1))
        y2 = max(0, min(y_max, y2))
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = h

    # Start the course a bit after the spawn zone
    cur_x = spawn_x + m_to_idx(0.4)

    # Goal 0: on the flat approach
    goals[0] = [spawn_x - m_to_idx(0.5), mid_y]

    # Build 7 repeated segments: beam -> pit -> pad -> pit, with slight lateral shifts
    lateral_choices = [-m_to_idx(0.45), -m_to_idx(0.25), 0, m_to_idx(0.25), m_to_idx(0.45)]
    last_center_y = mid_y

    for i in range(7):
        # Alternate beam and pad as the core repeated skill
        if i % 2 == 0:
            seg_len = beam_len
            seg_w = beam_w
            seg_h = beam_height
        else:
            seg_len = pad_len
            seg_w = pad_w
            seg_h = pad_height

        # Mild lateral offset to force steering while staying realistic
        offset = random.choice(lateral_choices)
        center_y = int(np.clip(last_center_y + offset, seg_w // 2 + 1, y_max - seg_w // 2 - 2))
        last_center_y = center_y

        y1 = center_y - seg_w // 2
        y2 = center_y + seg_w // 2

        # Put a raised obstacle segment
        add_rect(cur_x, cur_x + seg_len, y1, y2, seg_h)

        # Put the goal in the center of the current segment
        goals[i + 1] = [cur_x + seg_len / 2, center_y]

        # Carve a pit after each segment to force a deliberate step/jump
        pit_len = gap + (2 if i < 4 else 4) + m_to_idx(0.15 * difficulty)
        pit_x1 = cur_x + seg_len
        pit_x2 = min(x_max, pit_x1 + pit_len)
        height_field[pit_x1:pit_x2, :] = pit_depth

        # Advance to the next segment
        cur_x = pit_x2

        # Stop if we are too close to the end; we'll place the final goal in the remaining flat zone
        if cur_x >= x_max - m_to_idx(1.0):
            break

    # Restore the final approach area to flat ground
    if cur_x < x_max:
        height_field[cur_x:, :] = 0.0

    # Final goal near the end on flat ground
    goals[-1] = [x_max - m_to_idx(0.7), mid_y]

    # Ensure any remaining unused goals are still valid if the loop broke early
    for j in range(1, 7):
        if goals[j, 0] == 0 and goals[j, 1] == 0:
            goals[j] = [max(spawn_x + m_to_idx(0.5), x_max - m_to_idx(2.0)), mid_y]

    return height_field, goals