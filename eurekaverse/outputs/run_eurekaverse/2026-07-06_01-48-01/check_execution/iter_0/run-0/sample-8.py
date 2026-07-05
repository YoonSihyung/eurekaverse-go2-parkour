import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight sequence of raised balance beams over pits, testing precise foot placement and long jumps."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Full terrain size: 12m x 4m at 5cm resolution
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    n_x = height_field.shape[0]
    n_y = height_field.shape[1]
    mid_y = n_y // 2

    # --- Course design ---
    # Repeated "balance beam" obstacles:
    # A narrow raised strip runs forward in the center, with deep pits on both sides.
    # The robot must travel straight while staying on the beam, then step/jump across short gaps.
    #
    # Skill tested: balance, straight-line precision, and recovery on narrow elevated footing.

    # Difficulty-scaled parameters
    beam_height = 0.03 + 0.12 * difficulty          # low at easy, higher at hard
    beam_width_m = 0.45 + 0.20 * difficulty        # narrow but still realistic
    beam_width = max(9, m_to_idx(beam_width_m))     # >= 0.45m, keep at least narrow obstacle width
    beam_gap_m = 0.15 + 0.45 * difficulty          # gaps become longer with difficulty
    beam_gap = max(3, m_to_idx(beam_gap_m))
    beam_len_m = 1.15 + 0.35 * difficulty          # each beam is long enough to stand on
    beam_len = max(16, m_to_idx(beam_len_m))

    # Keep the first 2 meters flat for spawning
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # Pits after the spawn region encourage staying on the beam
    height_field[spawn_end:, :] = -0.85 - 0.25 * difficulty

    # Helper for placing centered rectangular beam
    def add_beam(x_start, x_end, center_y, width_idx, h):
        half_w = width_idx // 2
        y1 = max(0, center_y - half_w)
        y2 = min(n_y, center_y + half_w + (width_idx % 2))
        x1 = max(0, x_start)
        x2 = min(n_x, x_end)
        height_field[x1:x2, y1:y2] = h
        return x1, x2, y1, y2

    # Goal placement helpers
    def set_goal(i, x_m, y_m):
        goals[i] = [np.clip(m_to_idx(x_m), 0, n_x - 1), np.clip(m_to_idx(y_m), 0, n_y - 1)]

    # Initial goal near spawn, centered
    set_goal(0, 1.5, width / 2.0)

    # Build 7 beam segments after spawn, with small gaps between them
    cur_x = spawn_end + m_to_idx(0.25)
    for i in range(7):
        # Small lateral jitter to force minor alignment, but keep it traversable
        lateral_shift_m = random.uniform(-0.12, 0.12) * (0.4 + 0.6 * difficulty)
        center_y = int(np.clip(mid_y + m_to_idx(lateral_shift_m), 1, n_y - 2))

        # Slight variation in beam height to create stepping onto raised elements
        h = beam_height * (0.85 + 0.3 * random.random())

        x1, x2, y1, y2 = add_beam(cur_x, cur_x + beam_len, center_y, beam_width, h)

        # Place goal near the front-middle of each beam
        goal_x = (x1 + x2) / 2.0
        goal_y = (y1 + y2) / 2.0
        set_goal(i + 1, goal_x * field_resolution, goal_y * field_resolution)

        # Continue after beam + gap
        cur_x = x2 + beam_gap

        # Stop if we run out of terrain; remaining goals will be placed near the end
        if cur_x >= n_x - m_to_idx(0.8):
            break

    # If there is remaining goal count, place them toward the end of the course
    # on the final beam or on the landing zone.
    last_idx = int(np.max(np.where(goals[:, 0] > 0)[0])) if np.any(goals[:, 0] > 0) else 0
    for i in range(last_idx + 1, 8):
        gx = min(n_x - 2, cur_x + m_to_idx(0.4 * (i - last_idx)))
        gy = mid_y
        set_goal(i, gx * field_resolution, gy * field_resolution)

    # Ensure the final section is navigable and not clipped oddly
    # Keep the very end flat ground so the robot can finish cleanly.
    finish_start = max(spawn_end + 1, n_x - m_to_idx(1.2))
    height_field[finish_start:, :] = 0.0

    # Re-apply any beam section overlapping the finish zone minimally if needed
    # but keep the finish area mostly flat.
    for i in range(8):
        goals[i, 0] = np.clip(goals[i, 0], 0, n_x - 1)
        goals[i, 1] = np.clip(goals[i, 1], 0, n_y - 1)

    return height_field, goals