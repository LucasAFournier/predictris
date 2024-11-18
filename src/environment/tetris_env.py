import numpy as np
from typing import Literal
from .tetrominos import TETROMINO_COLORS, TETROMINO_VALUES, WALL_KICKS_CCW, WALL_KICKS_CW, TETROMINO_NAMES, TETROMINO_SHAPES
import random


NONE = -1


class Tetromino():
    def __init__(self, name, pos, orientation):
        self.name = name
        self.value = TETROMINO_VALUES[name]
        self.color = TETROMINO_COLORS[name]
        self.pos = pos
        self.orientation = orientation % 4
    
    def get_shape(self, new_orientation=None):
        possible_shapes = TETROMINO_SHAPES[self.name]
        
        if new_orientation!=None:
            return possible_shapes[new_orientation]
        
        return possible_shapes[self.orientation]


class TetrisEnv:
    def __init__(self, grid_shape: tuple[int, int]):
        self.agents = {} # dictionnary of agents
        self.grid = np.full(shape=grid_shape, fill_value=NONE)

        self.agents = {}  # Dictionary for agents (player)
        self.tetro_bag = []  # Tetromino bag
        self.current_tetro = self.RMG()  # Randomly generate a tetromino

    def RMG(self):
        """Randomly generate a tetromino and return its name."""
        if not self.tetro_bag:
            self.tetro_bag = TETROMINO_NAMES
        
        tetro_name = random.choice(self.tetro_bag)
        self.tetro_bag.remove(tetro_name)
        start_pos = self.grid.shape[0] //2 , 0
        
        return Tetromino(tetro_name, start_pos, 0)

    def check_collision(self, new_pos=None, new_orientation=None, offset_y=0, expand=True):
        """Check if a tetromino collides with the grid or other blocks."""
        tetro_shape = self.current_tetro.get_shape(new_orientation)
        
        if new_pos is None:
            new_pos = self.current_tetro.pos

        no_collision = True
        for offset in tetro_shape:
            rel_x, rel_y = offset
        
            if type(rel_x) == float:
                add = 0.5
            else:
                add = 0
        
            x = int(new_pos[0] + rel_x)
            y = int(new_pos[1] + rel_y)
        
            try:
                if self.grid[x, y+1] != NONE:
                    no_collision = False
            except IndexError:
                no_collision = False

        return no_collision