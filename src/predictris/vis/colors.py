from matplotlib.colors import ListedColormap

COLORS = {
    0: (220, 220, 220), # Light gray
    1: (0, 0, 0), # Black
}

DECORATION_COLORS = {
    "grid": (255, 255, 255),  # White gridlines
    "view_border": (255, 255, 0),  # Yellow border for viewing window
}

def rgbint_to_rgb(color):
    return tuple([x / 255 for x in color])

CMAP = ListedColormap([rgbint_to_rgb(color) for color in COLORS.values()])