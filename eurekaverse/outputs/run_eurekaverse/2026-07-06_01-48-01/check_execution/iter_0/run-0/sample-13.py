import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight-line sequence of alternating balance beams and stepping stones over shallow pits."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Ensure spawn area is clear
    spawn_clear_x = m_to_idx(2.0)
    height_field[:spawn_clear_x, :] = 0.0

    # Skill focus:
    # The robot must repeatedly balance on narrow raised beams and then step across short stones
    # placed over negative pits. This tests precise foot placement, lateral stability, and
    # smooth forward progression without large turns.

    # Difficulty-scaled parameters
    beam_height = 0.08 + 0.18 * difficulty          # raised enough to matter, but still realistic
    beam_width_m = 0.45 + 0.15 * (1.0 - difficulty) # narrow, but within rare narrow-obstacle allowance
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(0.4))

    stone_size_m = 0.55 + 0.15 * (1.0 - difficulty)
    stone_size = max(m_to_idx(stone_size_m), m_to_idx(0.4))

    pit_depth = -(0.35 + 0.55 * difficulty)        # deeper pits at higher difficulty
    gap_m = 0.35 + 0.35 * difficulty
    gap = m_to_idx(gap_m)

    # Lateral offset range for alternating obstacles
    side_offset_m = 0.55
    side_offset = m_to_idx(side_offset_m)

    # Helper to clamp slices safely
    def clamp(a, lo, hi):
        return max(lo, min(hi, a))

    # Helper to place a beam centered at y_center
    def add_beam(x1, x2, y_center, height):
        half_w = beam_width // 2
        y1 = clamp(y_center - half_w, 0, W)
        y2 = clamp(y_center + half_w + 1, 0, W)
        x1 = clamp(x1, 0, L)
        x2 = clamp(x2, 0, L)
        height_field[x1:x2, y1:y2] = height

    # Helper to place a stepping stone block centered at y_center
    def add_stone(x1, x2, y_center, height):
        half_s = stone_size // 2
        y1 = clamp(y_center - half_s, 0, W)
        y2 = clamp(y_center + half_s + 1, 0, W)
        x1 = clamp(x1, 0, L)
        x2 = clamp(x2, 0, L)
        height_field[x1:x2, y1:y2] = height

    # Start the course after the spawn area
    cur_x = spawn_clear_x

    # Place initial goal near the end of the spawn zone
    goals[0] = [spawn_clear_x - m_to_idx(0.4), mid_y]

    # Build 7 obstacle segments: beam, stone, beam, stone...
    # Each segment gets a goal placed on its center.
    for i in range(7):
        # Alternating lateral offset to force subtle steering and foot placement changes
        offset_dir = -1 if i % 2 == 0 else 1
        y_center = mid_y + offset_dir * side_offset

        # Keep obstacle fully inside bounds
        y_center = clamp(y_center, beam_width // 2 + 1, W - beam_width // 2 - 2)

        seg_len = m_to_idx(0.85 + 0.15 * (1.0 - difficulty))
        seg_len = max(seg_len, m_to_idx(0.6))

        if i % 2 == 0:
            # Beam segment: narrow raised walkway
            add_beam(cur_x, cur_x + seg_len, y_center, beam_height)
            goals[i + 1] = [cur_x + seg_len // 2, y_center]
        else:
            # Stone segment: square stepping platform
            add_stone(cur_x, cur_x + seg_len, y_center, 0.0)
            goals[i + 1] = [cur_x + seg_len // 2, y_center]

        # Surrounding pit to encourage staying on top of the obstacle
        pit_start = clamp(cur_x - gap // 2, 0, L)
        pit_end = clamp(cur_x + seg_len + gap // 2, 0, L)
        height_field[pit_start:pit_end, :] = pit_depth

        # Restore obstacle surface after pit fill
        if i % 2 == 0:
            add_beam(cur_x, cur_x + seg_len, y_center, beam_height)
        else:
            add_stone(cur_x, cur_x + seg_len, y_center, 0.0)

        # Leave a short flat connector between obstacles, still over pit so stepping matters
        cur_x += seg_len + gap

    # Final goal and finish region
    finish_len = m_to_idx(1.2)
    finish_start = clamp(cur_x, 0, L)
    finish_end = clamp(cur_x + finish_len, 0, L)

    # Bring the finish back to flat ground
    height_field[finish_start:, :] = 0.0
    goals[-1] = [clamp(finish_start + finish_len // 2, 0, L - 1), mid_y]

    # Ensure the spawn area remains flat
    height_field[:spawn_clear_x, :] = 0.0

    # Clamp goals to valid bounds and convert to integer indices
    goals = np.clip(goals, [0, 0], [L - 1, W - 1]).astype(np.int16)

    return height_field, goals