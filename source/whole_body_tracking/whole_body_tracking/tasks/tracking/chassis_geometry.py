"""Shared chassis geometry for the X2 "walk onto chassis" task.

MuJoCo `size` uses half-extents. The base chassis uses half-extents
(0.21, 0.21, 0.08) at z=0.08, so its top is at z=0.16 (16 cm platform).
"""

# World-frame chassis placement (meters).
CHASSIS_POS_X = 0.52
CHASSIS_POS_Y = -0.33

# Base platform (full dimensions for IsaacLab CuboidCfg): 0.42 x 0.42 x 0.16.
CHASSIS_BASE_SIZE = (0.42, 0.42, 0.16)
CHASSIS_BASE_POS = (CHASSIS_POS_X, CHASSIS_POS_Y, 0.08)

# Raised middle ridge (full dimensions): 0.42 x 0.14 x 0.08.
CHASSIS_MIDDLE_SIZE = (0.42, 0.14, 0.16)
CHASSIS_MIDDLE_POS = (CHASSIS_POS_X, CHASSIS_POS_Y, 0.24)

# Target MuJoCo contact/friction settings mirrored in scene assets.
CHASSIS_STATIC_FRICTION = 1.0
CHASSIS_DYNAMIC_FRICTION = 1.0
CHASSIS_RESTITUTION = 0.0001