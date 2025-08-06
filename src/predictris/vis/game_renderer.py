import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from predictris.agent import Agent
from predictris.tetris import TetrisEnvironment

from .colors import DECORATION_COLORS, COLORS

BLOCK_SIZE = 30
FPS = 60
GRID_SIZE: tuple[int, int] = (3, 3)
    

class TetrisRenderer:
    """A Pygame-based renderer for the Tetris agent visualization."""
    
    def __init__(self, env: TetrisEnvironment, agent: Agent) -> None:
        self.env = env
        self.agent = agent
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GRID_SIZE[0] * BLOCK_SIZE, GRID_SIZE[1] * BLOCK_SIZE)
        )
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        
        self.key_action_map: dict[int, callable] = {
            pygame.K_LEFT: lambda: self.env.act(0),  # Up
            pygame.K_UP: lambda: self.env.act(1),    # Left
            pygame.K_DOWN: lambda: self.env.act(2),  # Right
            pygame.K_RIGHT: lambda: self.env.act(3), # Down
            pygame.K_SPACE: lambda: self.env.act(4), # Rotate
        }

        self.render()

    def rectangle(self, start_x, start_y, width, height):
        rect_start_x = start_x * BLOCK_SIZE
        rect_start_y = start_y * BLOCK_SIZE
        rect_width, rect_height = width * BLOCK_SIZE, height * BLOCK_SIZE

        return pygame.Rect(rect_start_x, rect_start_y, rect_width, rect_height)

    def render_grid(self):
        observation = self.env.vision(self.agent)
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                obs_index = y * self.GRID_SIZE[0] + x
                color = COLORS[observation[obs_index]]
                rect = self.rectangle(x, y, 1, 1)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(
                    self.screen, DECORATION_COLORS["grid"], rect, 1
                )

    def render(self):
        self.render_grid()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
