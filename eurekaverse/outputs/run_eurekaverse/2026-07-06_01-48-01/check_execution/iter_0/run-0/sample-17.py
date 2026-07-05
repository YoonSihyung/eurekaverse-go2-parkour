import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A straight obstacle course of alternating low hurdles and narrow balance beams over pits."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain constants
    n_x = height_field.shape[0]
    n_y = height_field.shape[1]
    mid_y = n_y // 2

    # Keep the first 2 meters flat for spawning.
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # Use the rest of the course to force traversal over repeated obstacles.
    # Skill tested: precise foot placement, stepping over small hurdles, and maintaining balance
    # on narrow beams while crossing pits.
    height_field[spawn_end:, :] = -0.35 - 0.25 * difficulty

    # Difficulty-scaled obstacle geometry.
    hurdle_h = 0.08 + 0.14 * difficulty         # low step height
    hurdle_l = m_to_idx(0.40 + 0.20 * difficulty)
    hurdle_w = m_to_idx(1.10 + 0.30 * difficulty)

    beam_h = 0.02 + 0.08 * difficulty           # slightly raised beam
    beam_l = m_to_idx(0.70 + 0.25 * difficulty)
    beam_w = m_to_idx(0.45 + 0.10 * difficulty) # narrow but allowed exception

    pit_depth = -0.40 - 0.35 * difficulty

    # Goal progression layout: alternating hurdle -> pit -> beam -> pit -> ...
    # The robot is instructed to go in a straight line between consecutive goals.
    x = spawn_end + m_to_idx(0.4)
    goal_idx = 0

    def add_block(x_start, x_end, y_center, y_width, height):
        """Add a rectangular obstacle centered on the path."""
        half_w = max(1, y_width // 2)
        y1 = max(0, y_center - half_w)
        y2 = min(n_y, y_center + half_w)
        x1 = max(0, x_start)
        x2 = min(n_x, x_end)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = height

    def add_gap(x_start, x_end):
        """Ensure a pit segment is clearly negative."""
        x1 = max(0, x_start)
        x2 = min(n_x, x_end)
        if x1 < x2:
            height_field[x1:x2, :] = pit_depth

    # Create 4 repeated obstacle modules, giving 8 goals total.
    for i in range(4):
        # Hurdle segment
        y_off = int(np.random.randint(-m_to_idx(0.25), m_to_idx(0.25) + 1))
        y_c = np.clip(mid_y + y_off, 0, n_y - 1)
        add_block(x, x + hurdle_l, y_c, hurdle_w, hurdle_h)
        goals[2 * i] = [x + hurdle_l // 2, y_c]

        x += hurdle_l

        # Small landing / gap after hurdle
        gap_len = m_to_idx(0.65 + 0.25 * difficulty)
        add_gap(x, x + gap_len)
        goals[2 * i + 1] = [min(x + gap_len // 2, n_x - 1), mid_y]
        x += gap_len

        # Beam segment
        y_off2 = int(np.random.randint(-m_to_idx(0.18), m_to_idx(0.18) + 1))
        y_c2 = np.clip(mid_y + y_off2, 0, n_y - 1)
        add_block(x, x + beam_l, y_c2, beam_w, beam_h)
        goals[2 * i + 1] = [x + beam_l // 2, y_c2]

        x += beam_l

        # Another gap before next module
        gap2_len = m_to_idx(0.75 + 0.20 * difficulty)
        add_gap(x, x + gap2_len)
        if 2 * i + 2 < 8:
            goals[2 * i + 2] = [min(x + gap2_len // 2, n_x - 1), mid_y]
        x += gap2_len

    # Ensure the remainder of the terrain is a flat pit so the course stays focused.
    if x < n_x:
        height_field[x:, :] = pit_depth

    # Clamp goals to terrain bounds and ensure they are integers.
    goals[:, 0] = np.clip(goals[:, 0], 0, n_x - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, n_y - 1)

    return height_field, goals