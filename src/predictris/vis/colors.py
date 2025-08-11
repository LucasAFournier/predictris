from matplotlib.colors import ListedColormap

COLORS = {
    0: (220, 220, 220),  # Light gray
    1: (0, 0, 0),  # Black
}

DECORATION_COLORS = {
    "grid": (255, 255, 255),
}


CMAP = ListedColormap(
    [tuple([rgb / 255 for rgb in color]) for color in COLORS.values()]
)
