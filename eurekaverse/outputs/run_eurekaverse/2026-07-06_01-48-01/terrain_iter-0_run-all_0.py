
import numpy as np
import random

def set_terrain(terrain, variation, difficulty):
    terrain_fns = [
        set_terrain_0,
        set_terrain_1,
        set_terrain_2,
        set_terrain_3,
        set_terrain_4,
        set_terrain_5,
        set_terrain_6,
        set_terrain_7,
        set_terrain_8,
        set_terrain_9,
        # INSERT TERRAIN FUNCTIONS HERE
    ]
    idx = int(variation * len(terrain_fns))
    height_field, goals = terrain_fns[idx](terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale, terrain.horizontal_scale, difficulty)
    terrain.height_field_raw = (height_field / terrain.vertical_scale).astype(np.int16)
    terrain.goals = goals
    return idx

def set_terrain_0(length, width, field_resolution, difficulty):
    """A sequence of raised balance beams with alternating lateral offsets and short gaps."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe zone
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Make the course a repeated "balance beam over pit" skill
    # Beam width stays challenging but realistic for a quadruped.
    beam_width_m = 0.55 + 0.15 * (1.0 - difficulty)   # wider at low difficulty
    beam_width = max(m_to_idx(beam_width_m), m_to_idx(0.4))

    # Beam height and gap length scale with difficulty
    beam_height = 0.03 + 0.14 * difficulty
    gap_len_m = 0.28 + 0.55 * difficulty
    gap_len = max(m_to_idx(gap_len_m), m_to_idx(0.4))

    # Beam segment length
    beam_len_m = 0.85 + 0.15 * (1.0 - difficulty)
    beam_len = max(m_to_idx(beam_len_m), m_to_idx(0.4))

    # Lateral offsets alternate left/right to force small turns between goals
    max_offset_m = 0.55
    offset_steps = [-0.45, 0.35, -0.20, 0.50, -0.35, 0.25, -0.55]
    offset_scale = 0.35 + 0.65 * difficulty

    # Place a pit after the spawn so the robot must get onto the first beam
    height_field[spawn_x:, :] = -0.9

    cur_x = spawn_x

    for i in range(7):
        # Alternate lateral position with difficulty-dependent amplitude
        y_offset = int(round(offset_steps[i] * offset_scale * m_to_idx(max_offset_m)))
        center_y = int(np.clip(mid_y + y_offset, beam_width // 2 + 1, W - beam_width // 2 - 2))

        # Ensure the beam fits inside the terrain bounds
        x1 = cur_x
        x2 = min(cur_x + beam_len, L - 1)
        y1 = max(center_y - beam_width // 2, 0)
        y2 = min(center_y + beam_width // 2 + 1, W)

        # Create the raised beam
        height_field[x1:x2, y1:y2] = beam_height

        # Put a goal near the middle of each beam
        goals[i] = [x1 + max((x2 - x1) // 2, 1), center_y]

        # Carve the next gap
        cur_x = x2 + gap_len
        if cur_x >= L:
            cur_x = L - 1
            break
        height_field[x2:cur_x, :] = -0.9

    # Final goal on the last beam/landing area
    final_x = min(cur_x, L - 2)
    final_y = int(np.clip(mid_y + int(round(offset_steps[6] * offset_scale * m_to_idx(max_offset_m))), 0, W - 1))
    goals[7] = [final_x, final_y]

    # If the final beam didn't reach the end, keep the remainder as flat ground after the last challenge
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Ensure the spawn region remains flat
    height_field[:spawn_x, :] = 0.0

    return height_field, goals

def set_terrain_1(length, width, field_resolution, difficulty):
    """A centered stepping-stone course with narrow raised pads over shallow pits to train precision foot placement and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe region
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # --- Course design ---
    # Repeating skill: precise stepping between a sequence of narrow raised pads
    # with shallow pits between them. The robot must stay centered and control
    # foot placement while transitioning across consistent obstacles.

    # Difficulty-dependent parameters
    pad_length_m = 0.65 - 0.15 * difficulty          # longitudinal pad size
    pad_width_m = 1.15 - 0.15 * difficulty           # lateral pad size (>= 1m)
    pit_depth = -0.18 - 0.45 * difficulty             # shallow to moderate pits
    gap_length_m = 0.55 + 0.35 * difficulty           # spacing between pads
    pad_height = 0.03 + 0.07 * difficulty             # small raised platform

    pad_length = max(m_to_idx(0.4), m_to_idx(pad_length_m))
    pad_width = max(m_to_idx(1.0), m_to_idx(pad_width_m))
    gap_length = max(m_to_idx(0.4), m_to_idx(gap_length_m))

    # Add a slight lateral offset pattern so the robot must keep correcting
    lateral_offsets_m = [0.0, 0.12, -0.10, 0.14, -0.12, 0.10, -0.08]
    lateral_offsets = [m_to_idx(v) for v in lateral_offsets_m]

    # Keep all obstacles within bounds
    min_y = 0
    max_y = W

    # Fill course area after spawn with pits by default
    height_field[spawn_x:, :] = pit_depth

    # Build 7 pads and use 8 goals: one before the first pad, then one per pad
    cur_x = spawn_x + m_to_idx(0.4)

    # Goal 0: at the end of the spawn region
    goals[0] = [spawn_x - m_to_idx(0.25), mid_y]

    def place_pad(x_start, y_center, length_idx, width_idx, z):
        """Places a rectangular stepping pad."""
        x_end = min(L, x_start + length_idx)
        half_w = width_idx // 2
        y1 = max(min_y, y_center - half_w)
        y2 = min(max_y, y_center + half_w)
        height_field[x_start:x_end, y1:y2] = z
        return x_end

    # Create sequence of pads and goals
    for i in range(7):
        y_center = int(np.clip(mid_y + lateral_offsets[i], pad_width // 2, W - pad_width // 2 - 1))
        x_start = cur_x
        x_end = place_pad(x_start, y_center, pad_length, pad_width, pad_height)

        # Put goal near the center/front half of the pad
        goal_x = min(L - 1, x_start + pad_length // 2)
        goals[i + 1] = [goal_x, y_center]

        # Move forward to next pit segment
        cur_x = x_end + gap_length

        # If we are close to the end, stop early and let the final goal be on flat ground
        if cur_x >= L - m_to_idx(1.0):
            break

    # Ensure the remaining terrain after the final obstacle is flat ground so the last approach is clean
    final_flat_start = min(L, int(cur_x))
    height_field[final_flat_start:, :] = 0.0

    # Final goal placed on the flat landing zone near the end of the field
    goals[-1] = [L - m_to_idx(0.7), mid_y]

    # Clamp goals to valid indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals

def set_terrain_2(length, width, field_resolution, difficulty):
    """A straight-line balance-and-hop course with alternating narrow raised beams over pits."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)), dtype=np.float32)
    goals = np.zeros((8, 2), dtype=np.float32)

    # -----------------------------
    # Course design:
    # The robot starts on flat ground, then must cross a sequence of narrow elevated beams
    # separated by negative pits. This tests precise foot placement, lateral balance,
    # and repeated stepping/jumping without turning.
    # -----------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep the first part flat for safe spawn
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # The rest of the course is a pit so the robot must stay on the raised features
    height_field[spawn_len:, :] = -0.9 - 0.3 * difficulty

    # Beam / gap parameters scaled by difficulty
    beam_length_m = 0.75 - 0.15 * difficulty
    beam_length = max(m_to_idx(beam_length_m), m_to_idx(0.45))

    gap_length_m = 0.35 + 0.55 * difficulty
    gap_length = max(m_to_idx(gap_length_m), m_to_idx(0.25))

    beam_width_m = 1.05 - 0.15 * difficulty
    beam_half_width = max(m_to_idx(beam_width_m / 2.0), m_to_idx(0.55))

    beam_height = 0.05 + 0.18 * difficulty

    # Slight lateral offset pattern to force balance without creating a turn
    offset_choices_m = [0.0, 0.12, -0.12, 0.18, -0.18]
    offset_choices = [m_to_idx(v) for v in offset_choices_m]

    def add_beam(x1, x2, center_y, height):
        """Adds a narrow raised beam on top of the pit."""
        y1 = max(0, center_y - beam_half_width)
        y2 = min(W, center_y + beam_half_width)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = height

    # First goal near the end of the spawn area
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    # Build 6 beams and use the final goal as a landing point after the last beam
    cur_x = spawn_len
    for i in range(6):
        y_offset = random.choice(offset_choices)
        beam_center_y = int(np.clip(mid_y + y_offset, beam_half_width, W - beam_half_width - 1))

        # Add a beam
        add_beam(cur_x, cur_x + beam_length, beam_center_y, beam_height)

        # Goal placed near the center of the current beam
        goals[i + 1] = [cur_x + beam_length / 2.0, beam_center_y]

        # Advance with a pit gap
        cur_x += beam_length + gap_length

    # Ensure there is a safe landing zone at the end
    landing_len = m_to_idx(0.8)
    landing_start = min(cur_x, L - landing_len)
    height_field[landing_start:, :] = 0.0
    height_field[landing_start:, :] = np.maximum(height_field[landing_start:, :], 0.0)

    # Final goal on the landing zone
    goals[7] = [min(L - 1, landing_start + landing_len // 2), mid_y]

    # Make sure spawn area remains flat and safe
    height_field[:spawn_len, :] = 0.0

    return height_field, goals

def set_terrain_3(length, width, field_resolution, difficulty):
    """A repeating raised balance beam and stepping-stone course that tests precision foot placement and straight-line jumping."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # -------------------------------------------------------------------------
    # Course design:
    # - A flat spawn zone for the first 2 meters.
    # - Then a repeated sequence of narrow raised beams separated by pits.
    # - Each beam is wide enough for a quadruped, but narrow enough to demand
    #   careful straight-line alignment and balance.
    # - Obstacles get slightly harder with difficulty by increasing height,
    #   reducing beam width, and widening the pits.
    # -------------------------------------------------------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep spawn area flat and obstacle-free
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Terrain parameters scaled by difficulty
    beam_length = m_to_idx(0.85 - 0.15 * difficulty)      # modest platform length
    beam_length = max(beam_length, m_to_idx(0.55))
    beam_width = m_to_idx(1.25 - 0.35 * difficulty)       # narrow but realistic
    beam_width = max(beam_width, m_to_idx(0.9))
    beam_height = 0.12 + 0.22 * difficulty                # clear step up/down
    pit_length = m_to_idx(0.35 + 0.45 * difficulty)      # gap between beams
    pit_length = max(pit_length, m_to_idx(0.25))

    # Lateral wobble of the beam centerline, but kept within bounds
    max_offset = m_to_idx(0.35)
    offset_choices = [-max_offset // 2, 0, max_offset // 2]

    def place_beam(x1, x2, y_center, height):
        """Places a rectangular raised beam centered at y_center."""
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = height

    # Set everything after spawn to a pit first, then place beams on top
    height_field[spawn_len:, :] = -0.55 - 0.15 * difficulty

    # Start just after spawn
    cur_x = spawn_len
    center_y = mid_y

    # First goal near the end of the spawn zone
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    # Build 7 traversable segments after the first goal, with 7 intermediate goals
    for i in range(7):
        # Slight center shifts to require alignment, but keep the path straight enough
        center_y = int(np.clip(center_y + random.choice(offset_choices), beam_width // 2, W - beam_width // 2 - 1))

        # Place beam segment
        place_beam(cur_x, cur_x + beam_length, center_y, beam_height)

        # Put goal near the middle of the beam segment
        goals[i + 1] = [cur_x + beam_length / 2, center_y]

        # Advance over the beam and the pit
        cur_x += beam_length + pit_length

        # If we reach the end, stop placing and keep remainder flat to allow completion
        if cur_x >= L - m_to_idx(1.0):
            break

    # If there is leftover terrain after the last obstacle, make it flat ground
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Ensure the final goal is on reachable flat terrain near the end
    goals[-1] = [min(L - m_to_idx(0.6), max(cur_x, spawn_len + m_to_idx(0.5))), center_y]

    # Clamp all goals to valid indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals

def set_terrain_4(length, width, field_resolution, difficulty):
    """A straight-line sequence of narrow balance beams separated by pits, testing precision foot placement and gap clearing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Quantized dimensions
    length_idx = height_field.shape[0]
    width_idx = height_field.shape[1]
    mid_y = width_idx // 2

    # Spawn-safe flat region
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Skill focus: repeated narrow beam crossing over pits
    # Difficulty increases beam width reduction and gap length.
    beam_width_m = 1.2 - 0.45 * difficulty      # stays challenging but realistic
    beam_width_m = max(0.45, beam_width_m)      # allow narrow obstacle in the rare exception range
    beam_width = max(1, m_to_idx(beam_width_m))

    beam_length_m = 1.15 - 0.15 * difficulty
    beam_length = max(1, m_to_idx(beam_length_m))

    gap_m = 0.35 + 0.65 * difficulty
    gap = max(1, m_to_idx(gap_m))

    beam_height = 0.18 + 0.18 * difficulty
    pit_depth = -1.0

    # Slight lateral offsets create a mild steering requirement without becoming a turn course.
    y_offsets_m = [0.0, 0.15, -0.15, 0.20, -0.20, 0.10, -0.10, 0.0]
    y_offsets = [int(round(o / field_resolution)) for o in y_offsets_m]

    # Helper to place a raised beam and surrounding pit
    def add_beam(x1, x2, y_center):
        y1 = max(0, y_center - beam_width // 2)
        y2 = min(width_idx, y1 + beam_width)
        x1c = max(0, x1)
        x2c = min(length_idx, x2)
        if x1c < x2c and y1 < y2:
            # Create pit around the beam to prevent stepping around it
            height_field[x1c:x2c, :] = pit_depth
            height_field[x1c:x2c, y1:y2] = beam_height

    # First goal near the end of the spawn region
    goals[0] = [spawn_x - m_to_idx(0.45), mid_y]

    cur_x = spawn_x + m_to_idx(0.3)

    # Place 6 beams for goals 1..6, then a final landing area for goal 7
    for i in range(6):
        y_center = int(np.clip(mid_y + y_offsets[i], beam_width // 2, width_idx - beam_width // 2 - 1))
        add_beam(cur_x, cur_x + beam_length, y_center)

        # Goal centered on each beam
        goals[i + 1] = [cur_x + beam_length / 2.0, y_center]

        # Advance with a gap
        cur_x += beam_length + gap

    # Final beam / finish platform
    final_y = int(np.clip(mid_y + y_offsets[6], beam_width // 2, width_idx - beam_width // 2 - 1))
    final_x1 = min(cur_x, length_idx - beam_length - 1)
    final_x2 = min(final_x1 + beam_length, length_idx)
    add_beam(final_x1, final_x2, final_y)

    # Last goal on the final beam
    goals[7] = [final_x1 + beam_length / 2.0, final_y]

    # Ensure the spawn zone remains flat and safe
    height_field[:spawn_x, :] = 0.0

    # Clamp all goal coordinates to valid indices
    goals[:, 0] = np.clip(goals[:, 0], 0, length_idx - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, width_idx - 1)

    return height_field, goals

def set_terrain_5(length, width, field_resolution, difficulty):
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

def set_terrain_6(length, width, field_resolution, difficulty):
    """A straight corridor of alternating balance beams and low pit crossings to test precise foot placement and jump timing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic dimensions
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep spawn area clear
    spawn_x = m_to_idx(2.0)
    height_field[:spawn_x, :] = 0.0

    # Course parameters: repeated narrow beams over pits
    # Difficulty increases beam narrowness, pit depth, and spacing
    beam_width_m = 0.70 - 0.20 * difficulty   # around robot body width, but still walkable
    beam_width = max(m_to_idx(0.40), m_to_idx(beam_width_m))
    beam_thickness = m_to_idx(0.18 - 0.05 * difficulty)  # longitudinal landing area
    beam_height = 0.08 + 0.18 * difficulty

    pit_depth = -(0.35 + 0.55 * difficulty)
    gap_len = m_to_idx(0.55 + 0.25 * difficulty)
    beam_gap = m_to_idx(0.45 + 0.15 * difficulty)

    # Small lateral offsets encourage precise alignment but keep straight-line goal progression
    lateral_choices = [
        0,
        m_to_idx(0.12),
        -m_to_idx(0.12),
        m_to_idx(0.20),
        -m_to_idx(0.20),
    ]

    def add_beam(x1, x2, y_center, h):
        """Add a raised rectangular beam."""
        half_w = beam_width // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = h

    def add_pit(x1, x2):
        """Add a flat pit section."""
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x2 > x1:
            height_field[x1:x2, :] = pit_depth

    # Build eight goal locations: spawn, then alternating beam centers and recovery points
    cur_x = spawn_x
    goals[0] = [spawn_x - m_to_idx(0.4), mid_y]

    # Start with a short flat runup before the first beam
    runup_end = min(L, cur_x + m_to_idx(0.6))
    height_field[cur_x:runup_end, :] = 0.0
    cur_x = runup_end

    for i in range(1, 8):
        if i % 2 == 1:
            # Raised beam segment
            y_off = lateral_choices[(i + int(difficulty * 10)) % len(lateral_choices)]
            y_c = int(np.clip(mid_y + y_off, beam_width // 2, W - beam_width // 2 - 1))

            x1 = cur_x
            x2 = min(L, cur_x + beam_thickness)
            add_beam(x1, x2, y_c, beam_height)

            # Goal centered on the beam
            goals[i] = [x1 + (x2 - x1) / 2.0, y_c]

            # Advance past beam and insert a pit
            cur_x = x2
            pit_end = min(L, cur_x + gap_len)
            add_pit(cur_x, pit_end)
            cur_x = pit_end

        else:
            # Recovery flat landing strip after the pit
            landing_len = m_to_idx(0.8 + 0.15 * difficulty)
            x1 = cur_x
            x2 = min(L, cur_x + landing_len)
            height_field[x1:x2, :] = 0.0

            goals[i] = [x1 + (x2 - x1) / 2.0, mid_y]

            cur_x = x2
            # Add another pit to keep the robot jumping between usable surfaces
            pit_end = min(L, cur_x + beam_gap)
            add_pit(cur_x, pit_end)
            cur_x = pit_end

    # Ensure the final region is usable and the course ends flat
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Clamp goals to valid bounds and ensure integer indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals

def set_terrain_7(length, width, field_resolution, difficulty):
    """A straight-line course of alternating raised balance rails and shallow gaps to test precision foot placement and jump timing."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # ----------------------------
    # Course design:
    # - A flat spawn zone for safety
    # - A sequence of narrow raised rails separated by shallow pits
    # - The rails are wide enough to stand on, but narrow enough to force careful foot placement
    # - Difficulty increases by making the rails a bit longer/shorter, pits deeper, and lateral placement slightly less forgiving
    # - The robot moves straight along x, staying near the center line in y
    # ----------------------------

    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Keep the first part flat so the robot does not spawn in an obstacle
    spawn_len = m_to_idx(2.0)
    height_field[:spawn_len, :] = 0.0

    # Course parameters, tuned to stay realistic for a quadruped
    # Rail width stays at least 1 m, as requested
    rail_width_m = 1.05 - 0.15 * difficulty
    rail_width = max(m_to_idx(rail_width_m), m_to_idx(1.0))

    # Rail height is modest so it can be stepped on, but still creates a precision task
    rail_height = 0.10 + 0.12 * difficulty

    # Pit depth makes the robot commit to staying on the rails
    pit_depth = -(0.15 + 0.25 * difficulty)

    # Each segment includes a rail and a gap
    rail_len_m = 0.85 - 0.15 * difficulty
    gap_len_m = 0.45 + 0.35 * difficulty
    rail_len = max(m_to_idx(rail_len_m), m_to_idx(0.4))
    gap_len = max(m_to_idx(gap_len_m), m_to_idx(0.4))

    # Small lateral variation to keep the course interesting without making it a turning course
    y_jitter_max = max(1, m_to_idx(0.12 + 0.10 * difficulty))

    # Start just after the spawn area
    cur_x = spawn_len + m_to_idx(0.4)

    # First goal near the start of the course
    goals[0] = [spawn_len - m_to_idx(0.4), mid_y]

    def place_rail(x1, x2, center_y):
        """Places a raised narrow rail and surrounds it with a pit."""
        half_w = rail_width // 2
        y1 = max(0, center_y - half_w)
        y2 = min(W, center_y + half_w + (rail_width % 2))
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = rail_height

    # Build 6 rail segments, with goals placed on each one
    for i in range(6):
        if cur_x >= L - m_to_idx(1.0):
            break

        # Slight random lateral offset, but keep the rails safely inside bounds
        dy = random.randint(-y_jitter_max, y_jitter_max)
        rail_center_y = int(np.clip(mid_y + dy, rail_width // 2 + 1, W - rail_width // 2 - 2))

        # Add pit before the rail
        pit_x1 = cur_x
        pit_x2 = min(L, cur_x + gap_len)
        height_field[pit_x1:pit_x2, :] = pit_depth

        # Place the rail itself
        rail_x1 = pit_x2
        rail_x2 = min(L, rail_x1 + rail_len)
        height_field[rail_x1:rail_x2, :] = pit_depth
        place_rail(rail_x1, rail_x2, rail_center_y)

        # Goal near the center of the rail
        gx = rail_x1 + (rail_x2 - rail_x1) // 2
        goals[i + 1] = [gx, rail_center_y]

        # Advance
        cur_x = rail_x2 + m_to_idx(0.35)

    # Fill remaining area after the last rail with flat ground so the final goal can be reached cleanly
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Final goal placed on safe flat ground after the last obstacle
    final_goal_x = min(L - 1, cur_x + m_to_idx(0.6))
    goals[-1] = [final_goal_x, mid_y]

    # Ensure spawn zone remains flat even if any earlier slice touched it
    height_field[:spawn_len, :] = 0.0

    return height_field, goals

def set_terrain_8(length, width, field_resolution, difficulty):
    """A straight slalom of raised curb-like beams separated by shallow pits for careful trotting and balance."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Terrain dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe flat zone
    spawn_end = m_to_idx(2.0)
    height_field[:spawn_end, :] = 0.0

    # After spawn, create a repeated pattern of narrow raised beams with pits between them.
    # This tests precision foot placement, lateral balance, and maintaining forward momentum.
    # Heights are kept modest to remain realistic for a quadruped.

    # Difficulty-scaled geometry
    beam_len_m = 1.15 - 0.25 * difficulty
    beam_len = max(m_to_idx(beam_len_m), m_to_idx(0.6))

    pit_len_m = 0.45 + 0.65 * difficulty
    pit_len = max(m_to_idx(pit_len_m), m_to_idx(0.35))

    beam_w_m = 1.05 - 0.15 * difficulty
    beam_w = max(m_to_idx(beam_w_m), m_to_idx(0.75))

    # Lateral offset magnitude increases with difficulty but stays within the 4 m course width
    max_offset_m = 0.55 + 0.45 * difficulty
    max_offset = m_to_idx(max_offset_m)

    # Beam height: small curb-like step, enough to matter but still realistic
    beam_h = 0.07 + 0.06 * difficulty

    # Keep all obstacles beyond spawn area
    cur_x = spawn_end + m_to_idx(0.3)

    # Precompute a slalom pattern of offsets
    offsets = []
    for i in range(7):
        sign = -1 if i % 2 == 0 else 1
        # Slightly varying offset with difficulty-scaled amplitude
        base = int(sign * (0.35 * W + 0.35 * max_offset))
        jitter = random.randint(-m_to_idx(0.08), m_to_idx(0.08))
        offsets.append(base + jitter)

    # Helper to add a beam centered at y_center
    def add_beam(x1, x2, y_center, width_idx, height):
        half_w = width_idx // 2
        y1 = max(0, y_center - half_w)
        y2 = min(W, y_center + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = height

    # Goal 0: at the end of spawn zone
    goals[0] = [spawn_end - m_to_idx(0.35), mid_y]

    # Build 6 slalom beams with pits between them
    for i in range(6):
        if cur_x >= L - m_to_idx(1.0):
            break

        y_center = int(np.clip(mid_y + offsets[i], 0, W - 1))
        x1 = cur_x
        x2 = min(cur_x + beam_len, L)

        # Raised beam
        add_beam(x1, x2, y_center, beam_w, beam_h)

        # Goal placed near the center of each beam, slightly forward to encourage traversal
        g_idx = i + 1
        if g_idx < 8:
            goals[g_idx] = [x1 + (x2 - x1) * 0.6, y_center]

        # Pit after the beam to force stepping onto the next beam rather than shortcutting
        pit_x1 = x2
        pit_x2 = min(x2 + pit_len, L)
        height_field[pit_x1:pit_x2, :] = -0.45 - 0.15 * difficulty

        cur_x = pit_x2

    # Fill any remaining course with flat ground at 0 to make the final approach safe
    if cur_x < L:
        height_field[cur_x:, :] = 0.0

    # Final goals: place them progressively toward the end of the course
    # Goal 7 is near the finish line on flat ground
    finish_x = L - m_to_idx(0.7)
    goals[6] = [min(finish_x - m_to_idx(1.0), L - 1), int(np.clip(mid_y + offsets[5], 0, W - 1))]
    goals[7] = [finish_x, mid_y]

    # Ensure all goals are within bounds and integer-like indices
    goals[:, 0] = np.clip(goals[:, 0], 0, L - 1)
    goals[:, 1] = np.clip(goals[:, 1], 0, W - 1)

    return height_field, goals

def set_terrain_9(length, width, field_resolution, difficulty):
    """A straight-line stepping-stone course with raised balance beams over shallow pits."""

    def m_to_idx(m):
        """Converts meters to quantized indices."""
        return np.round(m / field_resolution).astype(np.int16) if not (isinstance(m, list) or isinstance(m, tuple)) else [round(i / field_resolution) for i in m]

    # Terrain grid
    height_field = np.zeros((m_to_idx(length), m_to_idx(width)))
    goals = np.zeros((8, 2))

    # Basic dimensions in indices
    L = m_to_idx(length)
    W = m_to_idx(width)
    mid_y = W // 2

    # Spawn-safe flat region
    spawn_end = m_to_idx(2.0)  # must keep obstacles beyond this x-index
    height_field[:spawn_end, :] = 0.0

    # Skill focus: precise foot placement and controlled stepping across repeated narrow beams
    # The course uses repeated raised beams separated by shallow pits; the robot must stay centered
    # and traverse them in a mostly straight line.
    n_beams = 7  # 7 obstacle segments between 8 goals
    remaining_x = L - spawn_end - m_to_idx(1.0)
    beam_len = max(m_to_idx(0.7), remaining_x // n_beams)
    beam_len = min(beam_len, m_to_idx(1.2))

    # Beam width is challenging but still realistic for a quadruped
    beam_half_width_m = 0.22 + 0.08 * (1.0 - difficulty)  # total width ~0.44 to 0.60 m
    beam_half_width = max(4, m_to_idx(beam_half_width_m))

    # Beam height increases slightly with difficulty
    beam_height = 0.08 + 0.12 * difficulty

    # Pit depth; negative heights force actual jumping/stepping rather than walking off edges
    pit_depth = -(0.18 + 0.22 * difficulty)

    # Separation between beams grows with difficulty
    gap_len = m_to_idx(0.45 + 0.35 * difficulty)
    gap_len = max(gap_len, m_to_idx(0.35))

    # Slight alternating lateral offsets to require small heading corrections, but still mostly straight
    y_offsets_m = [0.0, 0.12, -0.12, 0.18, -0.18, 0.10, -0.10]
    y_offsets = [m_to_idx(v) for v in y_offsets_m]

    def add_beam(x1, x2, cy, half_w, h):
        """Adds a rectangular beam centered at cy."""
        y1 = max(0, cy - half_w)
        y2 = min(W, cy + half_w)
        x1 = max(0, x1)
        x2 = min(L, x2)
        if x1 < x2 and y1 < y2:
            height_field[x1:x2, y1:y2] = h

    # Set pits in the active region first; beams will overwrite them
    height_field[spawn_end:, :] = pit_depth

    cur_x = spawn_end + m_to_idx(0.25)

    # First goal near the end of the spawn-flat region
    goals[0] = [spawn_end - m_to_idx(0.4), mid_y]

    # Place 7 beams and 7 intermediate goals
    for i in range(n_beams):
        cy = int(np.clip(mid_y + y_offsets[i], beam_half_width + 1, W - beam_half_width - 2))
        x1 = cur_x
        x2 = min(cur_x + beam_len, L)

        # Add beam
        add_beam(x1, x2, cy, beam_half_width, beam_height)

        # Goal at center of each beam
        goals[i + 1] = [(x1 + x2) / 2.0, cy]

        # Advance past beam and gap
        cur_x = x2 + gap_len

    # Final goal placed on flat ground after the last beam
    final_x = min(cur_x + m_to_idx(0.6), L - 1)
    final_y = int(np.clip(mid_y, 0, W - 1))
    goals[-1] = [final_x, final_y]

    # Ensure landing zone at the end is flat ground
    landing_start = min(cur_x, L)
    height_field[landing_start:, :] = 0.0

    # Also keep a narrow corridor around each beam from neighboring pit irregularities
    for i in range(n_beams):
        cy = int(np.clip(mid_y + y_offsets[i], beam_half_width + 1, W - beam_half_width - 2))
        x1 = spawn_end + m_to_idx(0.25) + i * (beam_len + gap_len)
        x2 = min(x1 + beam_len, L)
        y1 = max(0, cy - beam_half_width - 2)
        y2 = min(W, cy + beam_half_width + 2)
        height_field[x1:x2, y1:y2] = np.maximum(height_field[x1:x2, y1:y2], beam_height)

    return height_field, goals

# INSERT TERRAIN FUNCTION DEFINITIONS HERE