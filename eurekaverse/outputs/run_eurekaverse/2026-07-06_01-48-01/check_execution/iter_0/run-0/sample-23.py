import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight-line slalom of raised stepping pads with narrow lateral offsets and pits to test precise foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)

    # Basic course geometry
    spawn_len = m_to_idx(2.0)  # keep the first 2m flat and obstacle-free
    mid_y = W // 2

    # Skill focus: repeated lateral precision over raised pads separated by pits
    # Difficulty increases pad height, gap length, and sideways offset range
    pad_len = m_to_idx(0.85 - 0.15 * difficulty)          # 0.70m -> 0.85m
    pad_w = m_to_idx(1.25 - 0.10 * difficulty)            # wide enough for the quadruped
    gap_len = m_to_idx(0.55 + 0.75 * difficulty)          # larger gaps at higher difficulty
    pit_depth = -(0.35 + 0.45 * difficulty)               # negative terrain to discourage shortcuts
    pad_height = 0.04 + 0.22 * difficulty                 # modest step up, more challenging with difficulty

    # Side-to-side slalom offset: keep within bounds and realistic for a quadruped
    max_offset = m_to_idx(0.75 + 0.45 * difficulty)       # meters
    min_y = m_to_idx(0.6)
    max_y = W - m_to_idx(0.6)

    # Clear spawn zone
    height_field[:spawn_len, :] = 0.0

    # Fill the course after spawn with a pit, so the robot must use the pads
    height_field[spawn_len:, :] = pit_depth

    def place_pad(x1, x2, cy, h):
        """Place a rectangular pad centered at cy."""
        half_w = pad_w // 2
        y1 = max(0, cy - half_w)
        y2 = min(W, cy + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = h

    # Create 7 pads after the spawn, with goals on each pad plus final goal after the last one
    cur_x = spawn_len
    cur_y = mid_y

    # First goal near the end of the spawn area
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    for i in range(7):
        # Alternate the lateral direction for a slalom-like path
        direction = -1 if i % 2 == 0 else 1
        if i == 0:
            dy = 0
        else:
            # Smaller offsets early, larger later as difficulty increases
            dy = direction * random.randint(max(1, max_offset // 3), max_offset)
        cur_y = int(np.clip(cur_y + dy, min_y + pad_w // 2, max_y - pad_w // 2))

        x1 = cur_x
        x2 = cur_x + pad_len

        # Slightly vary pad height while keeping it climbable
        h = pad_height + np.random.uniform(-0.02, 0.03)
        place_pad(x1, x2, cur_y, h)

        # Put the goal near the center of the pad
        goals[i + 1] = [x1 + pad_len / 2, cur_y]

        # Advance to next pad start with a pit between
        cur_x = x2 + gap_len

        # Keep the next segment pit-filled unless it will be overwritten by a pad
        if cur_x < L:
            fill_end = min(L, cur_x)
            height_field[x2:fill_end, :] = pit_depth

    # Ensure the remaining tail is pit or flat depending on available space,
    # and place the final goal just beyond the last pad if possible.
    if cur_x < L:
        height_field[cur_x:, :] = pit_depth
        final_x = min(L - 1, cur_x + m_to_idx(0.45))
    else:
        final_x = L - 1

    # Final goal on the line after the last pad; keeps the robot moving forward
    goals[-1] = [final_x, cur_y]

    # Safety clamp to ensure all goals are inside bounds
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals