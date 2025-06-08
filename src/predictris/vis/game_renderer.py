import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment

from .colors import DECORATION_COLORS, COLORS
from .display import BLOCK_SIZE, FPS


class TetrisRenderer:
    """A Pygame-based renderer for the Tetris environment visualization."""
    
    GRID_SIZE: tuple[int, int] = (11, 11)
    
    def __init__(self, env: TetrisEnvironment, agent: Agent) -> None:
        """Initialize the Tetris renderer with a Pygame window.
        
        Args:
            env: The Tetris environment to render
            agent: The agent controlling the game
        """
        self.env = env
        self.agent = agent
        self.center = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        
        # Initialize display
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.GRID_SIZE[0] * BLOCK_SIZE, self.GRID_SIZE[1] * BLOCK_SIZE)
        )
        pygame.display.set_caption("Predictris")
        self.clock = pygame.time.Clock()
        
        # Event mapping for manual control
        self.key_action_map: dict[int, callable] = {
            pygame.K_LEFT: env.move_tetromino_left,
            pygame.K_RIGHT: env.move_tetromino_right,
            pygame.K_DOWN: env.move_tetromino_down,
            pygame.K_UP: env.move_tetromino_up,
            pygame.K_SPACE: env.rotate_tetromino_cw,
        }

        self.render()

    def rectangle(self, start_x, start_y, width, height):
        """Generate rectangle from environment coordinates"""
        rect_start_x = start_x * BLOCK_SIZE
        rect_start_y = start_y * BLOCK_SIZE
        rect_width, rect_height = width * BLOCK_SIZE, height * BLOCK_SIZE

        return pygame.Rect(rect_start_x, rect_start_y, rect_width, rect_height)

    def render_grid(self):
        """Render the Tetris grid."""
        for x in range(0, self.GRID_SIZE[0]):
            for y in range(0, self.GRID_SIZE[1]):
                rect = self.rectangle(x, y, 1, 1)
                pygame.draw.rect(
                    self.screen, COLORS[0], rect
                )  # Draw background to erase previous blocks
                pygame.draw.rect(
                    self.screen, DECORATION_COLORS["grid"], rect, 1
                )  # Draw grid lines

    def render_tetromino(self):
        """Render the tetromino."""
        offset = self.env.offset
        pos = self.env.position

        for dx, dy in self.env.get_shape():
            x, y = self.center[0] + pos[0] + int(dx + offset), self.center[1] + pos[1] + int(dy + offset)
            rect = self.rectangle(x, y, 1, 1)
            pygame.draw.rect(self.screen, COLORS[1], rect)
            pygame.draw.rect(
                self.screen, DECORATION_COLORS["grid"], rect, 1
            )  # Border for tetromino blocks

    def render_view_border(self):
        """
        Render the viewing window of the AgentBody object.
        """
        rect = self.rectangle(self.center[0] - 1, self.center[1] - 1, 3, 3)
        pygame.draw.rect(
            self.screen, DECORATION_COLORS["view_border"], rect, 2
        )  # Thicker border

    def render(self):
        """Render the entire game state"""
        self.render_grid()
        self.render_tetromino()
        self.render_view_border()

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()
