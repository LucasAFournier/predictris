# Tetromino names and colors
TETROMINO_NAMES = ["T", "J", "I", "O", "L", "Z", "S"]

TETROMINO_VALUES = {
    "T": 0,
    "J": 1,
    "I": 2,
    "O": 3,
    "L": 4,
    "Z": 5,
    "S": 6,    
}

TETROMINO_COLORS = {
    "T": "magenta",
    "J": "blue",
    "I": "cyan",
    "O": "yellow",
    "L": "orange",
    "Z": "red",
    "S": "green",
}

# Tetromino definitions with rotation states and offsets
TETROMINO_SHAPES = {
    "T": [
        [(-1, 0), (0, 0), (1, 0), (0, -1)], # list of offsets wrt center with 0° rotation
        [(0, 1), (0, 0), (1, 0), (0, -1)], # " " with 90° rotation
        [(-1, 0), (0, 0), (1, 0), (0, 1)], # " " with 180° rotation
        [(-1, 0), (0, 0), (0, 1), (0,-1)], # " " with 270° rotation
        1.0, 1.5 # width & height
    ],
    "J": [
        [(-1, 0), (0, 0), (1, 0), (-1, -1)],
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, 1)],
        1.0, 1.5
    ],
    "I": [
        [(-1.5, -0.5), (-0.5, -0.5), (0.5, -0.5), (1.5, -0.5)],
        [(0.5, -0.5),  (0.5, 0.5),  (0.5, 1.5),  (0.5, -1.5)],
        [(-1.5,  0.5), (-0.5,  0.5), (0.5,  0.5), (1.5,  0.5)],
        [(-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5), (-0.5, -1.5)],
        1.0, 1.5
    ],
    "O": [
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)],
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)],
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)],
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)],
        1.0, 1.0
    ],
    "L": [
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, -1)],
        1.0, 1.5
    ],
    "Z": [
        [(-1, -1), (0, -1), (0, 0), (1, 0)],
        [(1, -1), (1, 0), (0, 0), (0, 1)],
        [(1, 1), (0, 1), (0, 0), (-1, 0)],
        [(-1, 1), (-1, 0), (0, 0), (0, -1)],
        1.0, 1.5
    ],
    "S": [
        [(0, 0), (-1, 0), (0, -1), (1, -1)],
        [(0, 0), (0, -1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1)],
        [(0, 0), (0, 1), (-1, 0), (-1, -1)],
        1.0, 1.5
    ],
}

# Wall kick data: Adjustments for successful rotation near walls.
# These offsets are applied during a rotation to prevent collisions.
WALL_KICKS_CW = [ # Clockwise wall kicks
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],   # 0° -> 90°
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],     # 90° -> 180°
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],      # 180° -> 270°
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],  # 270° -> 0°
]
WALL_KICKS_CCW = [ # Counterclockwise wall kicks
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],      # 0° -> 270°
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],  # 90° -> 0°
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],   # 180° -> 90°
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],     # 270° -> 180°
]