import numpy as np
from typing import Optional
from ..utils import (
    NONE,
    TETROMINO_NAMES_TO_VALUES,
    TETROMINO_NAMES,
    TETROMINO_SHAPES,
    # WALL_KICKS_CCW,
    # WALL_KICKS_CW,
)
import random

class Tetromino:
    def __init__(
        self,
        name: str,
        pos: tuple[int, int],
        orientation: int
    ):
        """Initialize a Tetromino piece."""    
        self.name = name    
        self.pos = pos  # (x, y) position on the grid
        self.orientation = orientation
        self.shapes = TETROMINO_SHAPES[name]
        self.value = TETROMINO_NAMES_TO_VALUES[name]
        self.offset = TETROMINO_SHAPES[name][-1]
    
    def get_shape(self, new_orientation: Optional[int] = None):
        if new_orientation is None:
            return self.shapes[self.orientation]
        else:
            return self.shapes[new_orientation]


class TetrisEnv:
    def __init__(
            self,
            grid_shape: Optional[tuple[int, int]] = None,
            start_grid: Optional[np.ndarray] = None,
            start_tetromino: Optional[Tetromino] = None,
        ):
        """Initialize the Tetris environment."""
        if start_grid is not None:
            self.grid = start_grid # useless for the moment
        elif grid_shape:
            self.grid = np.full(shape=grid_shape, fill_value=NONE, dtype=int)  # Game grid
        else:
            raise Exception("Either a grid shape or an already defined start grid must be defined to initialize environment.")
        
        if start_tetromino is None:
            self.current_tetromino = self.generate_tetromino()  # Current active Tetromino
        elif start_tetromino.name in TETROMINO_NAMES:
            self.current_tetromino = start_tetromino
        else:
            raise Exception(f"{start_tetromino.name} is not a valid tetromino name. Should be one of {TETROMINO_NAMES}")

    def get_state(self):
        """Create a compact representation of the environment state."""
        return {
            "grid": self.grid.copy(),
            "tetromino_name": self.current_tetromino.name,
            "tetromino_pos": self.current_tetromino.pos,
            "tetromino_orientation": self.current_tetromino.orientation,
        }

    @classmethod
    def from_state(cls, state: dict):
        """Reconstruct the environment from its state."""
        start_tetromino = Tetromino(state["tetromino_name"], state["tetromino_pos"], state["tetromino_orientation"])
        env = cls(
            start_grid=state["grid"],
            start_tetromino=start_tetromino,
        )
        return env

    def generate_tetromino(self) -> Tetromino:
        """
        Generate a new Tetromino using the 7-bag randomization method.
        Replenishes the bag if empty.
        """
        name = random.choice(TETROMINO_NAMES)
        print(name)
        start_pos = (self.grid.shape[0] // 2, self.grid.shape[1] // 2)
        start_orientation = 0
        return Tetromino(name, start_pos, start_orientation)

    def is_collision(
        self,
        new_pos: Optional[tuple[int, int]] = None,
        new_orientation: Optional[int] = None
    ) -> bool:
        """
        Check if the current Tetromino would collide with the grid or other blocks.
        """
        if new_pos is None:
            new_pos = self.current_tetromino.pos
        if new_orientation is None:
            new_orientation = self.current_tetromino.orientation

        tetro_shape = self.current_tetromino.get_shape(new_orientation)
        offset = self.current_tetromino.offset

        for dx, dy in tetro_shape:
            x = int(new_pos[0] + dx + offset)
            y = int(new_pos[1] + dy + offset)
            
            # Check bounds and collisions
            if x < 0 or x >= self.grid.shape[0] or y < 0 or y >= self.grid.shape[1]:
                return True  # Out of bounds
            if self.grid[x, y] != NONE:
                return True  # Collision with existing block

        return False  # No collision

    # def place_tetromino(self) -> None:
    #     """
    #     Place the current Tetromino on the grid and lock it in place.
    #     """
    #     tetro_shape = self.current_tetromino.get_shape()
    #     for offset_x, offset_y in tetro_shape:
    #         x = int(self.current_tetromino.pos[0] + offset_x)
    #         y = int(self.current_tetromino.pos[1] + offset_y)
    #         self.grid[x, y] = self.current_tetromino.value

    #     # Generate a new Tetromino
    #     self.current_tetromino = self.generate_tetromino()