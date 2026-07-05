import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight corridor of alternating low hurdles and narrow bridge gaps to test rhythmic stepping and controlled jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe flat zone
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # Skill focus: repeated low hurdles with occasional short pits between them.
    # The robot must maintain forward momentum and step/jump consistently.
    #
    # Obstacles are kept within bounds and placed only after the spawn region.

    # Difficulty-scaled parameters
    hurdle_h = 0.08 + 0.18 * difficulty          # 8 cm to 26 cm
    hurdle_l = 0.45 - 0.10 * difficulty          # 45 cm to 35 cm
    hurdle_l = max(0.35, hurdle_l)

    bridge_w = 0.60 - 0.15 * difficulty          # 60 cm to 45 cm
    bridge_w = max(0.40, bridge_w)

    gap_l = 0.40 + 0.55 * difficulty             # 40 cm to 95 cm
    gap_l = max(0.30, gap_l)

    corridor_half_w = m_to_idx(0.85)             # ~1.7 m wide corridor
    y1 = max(0, mid_y - corridor_half_w)
    y2 = min(W, mid_y + corridor_half_w)

    # A narrow central lane for the bridge segments
    bridge_half_w = m_to_idx(bridge_w / 2.0)

    # Helper to clamp slices
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # Starting goal just after spawn
    goals[0] = [spawn_end - m_to_idx(0.25), mid_y]

    cur_x = spawn_end

    # Create 7 sequential segments so we can place 8 goals total
    # Pattern: hurdle -> gap -> hurdle -> gap ... with slight variation
    for i in range(7):
        # Small random variation to prevent overfitting to exact distances
        x_jitter = random.randint(-m_to_idx(0.08), m_to_idx(0.08))
        seg_start = clamp(cur_x + x_jitter, spawn_end, L - 2)
        seg_end = clamp(seg_start + m_to_idx(hurdle_l), seg_start + 1, L)

        # Alternate between a full-width low hurdle and a narrower bridge-like step
        if i % 2 == 0:
            # Low hurdle across the corridor width
            height_field[seg_start:seg_end, y1:y2] = hurdle_h
            goal_x = seg_start + (seg_end - seg_start) // 2
            goal_y = mid_y
        else:
            # Narrow bridge in the center, with negative sides to encourage staying centered
            pit_depth = -0.35 - 0.35 * difficulty
            height_field[seg_start:seg_end, y1:y2] = pit_depth
            b1 = clamp(mid_y - bridge_half_w, 0, W)
            b2 = clamp(mid_y + bridge_half_w, 0, W)
            height_field[seg_start:seg_end, b1:b2] = hurdle_h
            goal_x = seg_start + (seg_end - seg_start) // 2
            goal_y = mid_y

        # Goal near the center of the active traversal feature
        goals[i + 1] = [goal_x, goal_y]

        # Advance past the obstacle and a gap
        cur_x = seg_end + m_to_idx(gap_l)

        # Fill the gap area with a pit to prevent simply walking around the feature
        pit_start = seg_end
        pit_end = clamp(cur_x, pit_start, L)
        if pit_end > pit_start:
            height_field[pit_start:pit_end, y1:y2] = -0.45 - 0.25 * difficulty

    # Final goal near the end of the course
    end_x = clamp(L - m_to_idx(0.6), spawn_end, L - 1)
    goals[-1] = [end_x, mid_y]

    # Ensure the landing / finish zone is flat and safe
    finish_start = clamp(L - m_to_idx(1.0), 0, L)
    height_field[finish_start:, :] = 0.0

    return height_field, goals