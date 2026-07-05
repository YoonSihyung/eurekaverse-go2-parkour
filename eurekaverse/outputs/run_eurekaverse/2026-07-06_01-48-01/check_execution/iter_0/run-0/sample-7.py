import numpy as np
import random

def set_terrain(length, width, field_resolution, difficulty):
    """A repeating series of narrow stepping platforms with alternating lateral offsets over a pit."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain dimensions in cells
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe flat region
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Difficulty-scaled obstacle parameters
    # Platforms stay within the width bound while still requiring accurate foot placement.
    platform_len = m_to_idx(0.75 - 0.15 * difficulty)   # along x
    platform_wid = m_to_idx(1.00 - 0.20 * difficulty)   # along y, but never too narrow
    platform_h = 0.10 + 0.18 * difficulty               # elevated above the pit

    # Gaps widen with difficulty to make jumping/stepping more demanding.
    gap_len = m_to_idx(0.35 + 0.55 * difficulty)

    # Lateral offsets alternate to force slight turns in the goal sequence.
    max_offset = m_to_idx(0.55)
    offset_mag = int(round((0.15 + 0.85 * difficulty) * max_offset))
    offset_mag = max(0, min(offset_mag, max_offset))

    # Put the rest of the course into a pit so the robot must use the platforms.
    height_field[spawn_len:, :] = -1.0

    def add_platform(x_center, y_center):
        """Adds a rectangular platform and clips it to the terrain bounds."""
        x1 = max(0, x_center - platform_len // 2)
        x2 = min(L, x_center + (platform_len - platform_len // 2))
        y1 = max(0, y_center - platform_wid // 2)
        y2 = min(W, y_center + (platform_wid - platform_wid // 2))
        height_field[x1:x2, y1:y2] = platform_h

    # First goal near the end of the spawn area
    goals[0] = [spawn_len - m_to_idx(0.35), mid_y]

    # Build a consistent sequence of 7 platforms after the spawn
    cur_x = spawn_len + m_to_idx(0.25)
    direction = 1

    for i in range(7):
        # Alternate sideways placement to require minor heading changes
        y_center = mid_y + direction * offset_mag
        y_center = int(np.clip(y_center, platform_wid // 2, W - 1 - platform_wid // 2))

        add_platform(cur_x, y_center)

        # Goal sits near the center of each platform
        goals[i + 1] = [cur_x, y_center]

        # Advance to next platform position
        cur_x += platform_len + gap_len

        # Alternate the offset direction for a zig-zag stepping pattern
        direction *= -1

    # Ensure the terrain beyond the final platform remains flat pit-free only if needed for last goal visibility.
    # Keep it as pit to preserve the jumping/stepping challenge.
    height_field[:spawn_len, :] = 0.0

    return height_field, goals