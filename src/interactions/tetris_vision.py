from typing import Optional
import numpy as np

from ..environment.tetris_env import TetrisEnv


class TetrisVision:
    def __init__(
        self,
        env: TetrisEnv,
        view_radius: int,
        view_pos: Optional[tuple[int, int]] = None,
        view_range_outside_grid: Optional[int] = None,
    ):
        """
        Initialize the TetrisVision object, which represents the agent's view of the Tetris grid.
        """
        self.env = env
        self.view_radius = view_radius
        self.view_shape = (2 * view_radius + 1, 2 * view_radius + 1)
        
        self.view_range_outside_grid = (
            view_range_outside_grid if view_range_outside_grid is not None else view_radius
        )

        self.view_pos = view_pos or (env.grid.shape[0] // 2, env.grid.shape[1] // 2)
        
        diff = self.view_range_outside_grid - self.view_radius
        grid_w, grid_h = self.env.grid.shape
        min_x, min_y = -diff, -diff
        max_x, max_y = grid_w + diff - 1, grid_h + diff - 1
        self.view_pos_bounds = (min_x, max_x, min_y, max_y)

    def get_state(self):
        """Create a compact representation of the vision state."""
        return {
            "view_radius": self.view_radius,
            "view_range_outside_grid": self.view_range_outside_grid,
            "view_pos": self.view_pos,
        }

    @classmethod
    def from_state(cls, state: dict, env: TetrisEnv):
        """Reconstruct the vision object from its state."""
        return cls(
            env=env,
            view_radius=state["view_radius"],
            view_range_outside_grid=state["view_range_outside_grid"],
            view_pos=state["view_pos"],
        )

    def is_inside_viewing_border(self, new_view_pos: tuple[int, int]) -> bool:
        """
        Check if the proposed viewing position exceeds the grid plus the allowable range outside.
        """
        x, y = new_view_pos
        min_x, max_x, min_y, max_y = self.view_pos_bounds
        
        return (min_x <= x <= max_x and min_y <= y <= max_y)

    def observe(self) -> np.ndarray:
        """
        Generate a view of the grid centered on the current viewing position,
        including the current Tetromino.
        """
        grid_h, grid_w = self.env.grid.shape
        view_h, view_w = self.view_shape
        center_x, center_y = self.view_pos

        # Create an empty view with a padding value
        padded_view = np.full(self.view_shape, fill_value=-1, dtype=int)

        # Determine view bounds on the grid
        start_x = max(0, center_x - self.view_radius)
        end_x = min(grid_h, center_x + self.view_radius + 1)
        start_y = max(0, center_y - self.view_radius)
        end_y = min(grid_w, center_y + self.view_radius + 1)

        # Map to the padded view
        view_start_x = start_x - (center_x - self.view_radius)
        view_start_y = start_y - (center_y - self.view_radius)
        padded_view[
            view_start_x:view_start_x + (end_x - start_x),
            view_start_y:view_start_y + (end_y - start_y)
        ] = self.env.grid[start_x:end_x, start_y:end_y]

        # Add the Tetromino to the view
        tetro = self.env.current_tetromino
        value = tetro.value
        for dx, dy in tetro.get_shape():
            x, y = tetro.pos[0] + int(dx), tetro.pos[1] + int(dy)
            rel_x, rel_y = x - start_x + view_start_x, y - start_y + view_start_y

            if 0 <= rel_x < view_h and 0 <= rel_y < view_w:
                padded_view[rel_x, rel_y] = value

        return padded_view.flatten()
