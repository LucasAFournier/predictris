import pygame
import numpy as np
from typing import Optional

from ..environment import TetrisEnv
from ..interactions import TetrisVision
from ..utils import NONE, TETROMINO_VALUES_TO_COLORS

# Define constants for rendering
COLORS = {
    "grid": (200, 200, 200), # Light gray gridlines
    "view border": (255, 255, 0),  # Yellow border for viewing window
}
BLOCK_SIZE = 30  # Size of each block in pixels
FPS = 60  # Frames per second


class TetrisRenderer:
    def __init__(self, env: TetrisEnv, vision: Optional[TetrisVision] = None):
        """
        Initialize the Tetris renderer with a Pygame window.
        :param env: TetrisEnv instance to visualize.
        """
        self.env = env
        self.vision = vision

        # Initialize Pygame
        pygame.init()
        grid_w, grid_h = env.grid.shape[0], env.grid.shape[1]
        add = self.vision.view_range_outside_grid
        screen_w, screen_h = (grid_w + 2 * add) * BLOCK_SIZE, (grid_h + 2 * add) * BLOCK_SIZE
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()

        self.render()

    def rectangle(self, start_x, start_y, width, height):
        """Generate rectangle from environment coordinates"""
        add = self.vision.view_range_outside_grid
        rect_start_x = (start_x + add) * BLOCK_SIZE
        rect_start_y = (start_y + add) * BLOCK_SIZE
        rect_width, rect_height = width * BLOCK_SIZE, height * BLOCK_SIZE
        
        return pygame.Rect(rect_start_x, rect_start_y, rect_width, rect_height)

    def render_grid(self):
        """Render the Tetris grid."""
        for x in range(0, self.env.grid.shape[0]):
            for y in range(0, self.env.grid.shape[1]):
                rect = self.rectangle(x, y, 1, 1)
                pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)  # Draw grid lines

    def render_blocks(self):
        """Render the blocks currently in the grid."""
        for x in range(self.env.grid.shape[0]):
            for y in range(self.env.grid.shape[1]):
                value = self.env.grid[x, y]
                if value != NONE:
                    color = TETROMINO_VALUES_TO_COLORS[value]
                    rect = self.rectangle(x, y, 1, 1)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)  # Border for blocks

    def render_tetromino(self):
        """Render the active tetromino."""
        
        tetro = self.env.current_tetromino
        value = tetro.value
        offset = tetro.offset
        color = TETROMINO_VALUES_TO_COLORS[value]
        for dx, dy in tetro.get_shape():
            x, y = tetro.pos[0] + dx + offset, tetro.pos[1] + dy + offset
            rect = self.rectangle(x, y, 1, 1)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)  # Border for tetromino blocks

    def render_view(self):
        """
        Render the viewing window of the TetrisVision object.
        """
        center_x, center_y = self.vision.view_pos
        view_radius = self.vision.view_radius

        top_left_x = center_x - view_radius
        top_left_y = center_y - view_radius
        view_size = 2 * view_radius + 1
        
        rect = self.rectangle(top_left_x, top_left_y, view_size, view_size)
        pygame.draw.rect(self.screen, COLORS["view border"], rect, 2)  # Thicker border
        
    def render(self):
        """
        Render the entire game state, optionally including a viewing window.
        :param vision: Optional TetrisVision instance for rendering the viewing window.
        """
        self.screen.fill(TETROMINO_VALUES_TO_COLORS[NONE])
        self.render_grid()
        # self.render_blocks() # unused for the moment
        self.render_tetromino()
        if self.vision is not None:
            self.render_view()

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()
