import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import random

from predictris.agent import Agent
from predictris.learning import PredictionTree

from .constants import TETROMINO_SHAPES, TETRIS_ACTIONS, TETRIS_PERCEPTIONS, TETROMINO_NAMES


class TetrisEnvironment:
    """Represents the Tetris game environment."""

    def __init__(self, tetromino_state: dict = None):
        """Initialize the Tetris environment with a given Tetromino state.

        Args:
            tetromino_state (dict, optional): Dictionary representation of the Tetromino state. Defaults to None.
        """
        if tetromino_state is None:
            self.random_init()

        else:
            self.name = tetromino_state["name"]
            self.position = tetromino_state["position"]
            self.orientation = tetromino_state["orientation"]

        self.shapes = TETROMINO_SHAPES[self.name]
        self.offset = self.shapes["offset"]

    def random_init(self, tetrominos: list[str] = None):
        """Create a random environment with a tetromino placed near the view position."""
        if not tetrominos:
            tetrominos = TETROMINO_NAMES
        
        self.__init__(
            tetromino_state={
                "name": random.choice(tetrominos),
                "position": (random.randint(-3, 3), random.randint(-3, 3)),
                "orientation": random.randint(0, 3),
            }
        )
        
    def get_shape(self, orientation: int = None) -> list[tuple[int, int]]:
        """Returns the shape of the Tetromino in the given orientation.
        
        Args:
            orientation (int, optional): Orientation of the Tetromino. Defaults to None.

        Returns:
            list[tuple[int, int]]: List of (dx, dy) coordinates of the Tetromino shape
        """
        if orientation is None:
            orientation = self.orientation
        return self.shapes[orientation]

    #region Interactions

    def move_tetromino(self, dx: int, dy: int) -> None:
        """Move the current Tetromino by the specified delta coordinates.

        Args:
            dx (int): Horizontal movement (-1 for left, 1 for right)
            dy (int): Vertical movement (-1 for up, 1 for down)
        """
        self.position = (self.position[0] + dx, self.position[1] + dy)

    def move_tetromino_left(self, agent: Agent):
        self.move_tetromino(dx=-1, dy=0)

    def move_tetromino_right(self, agent: Agent):
        self.move_tetromino(dx=1, dy=0)

    def move_tetromino_down(self, agent: Agent):
        self.move_tetromino(dx=0, dy=1)

    def move_tetromino_up(self, agent: Agent):
        self.move_tetromino(dx=0, dy=-1)

    def rotate_tetromino(self, clockwise: bool = True):
        """Rotate the current Tetromino in the specified direction.

        Args:
            clockwise (bool, optional): Direction of rotation. Defaults to True.
        """
        rotation_delta = 1 if clockwise else -1
        self.orientation = (self.orientation + rotation_delta) % 4

    def rotate_tetromino_cw(self, _: int = None) -> None:
        self.rotate_tetromino(clockwise=True)

    def vision(self, agent: Agent) -> tuple:
        """Observe the current state of the environment.

        Args:
            agent (Agent): Agent object observing the environment.
        
        Returns:
            tuple: Flattened view matrix of the environment.
        """
        # tetro_x, tetro_y = self.pos
        # view_radius_x, view_radius_y = body.view_radius

        # view = np.full(
        #     shape=(2 * view_radius_x + 1, 2 * view_radius_y + 1),
        #     fill_value=BACKGROUND_VALUE,
        #     dtype=np.int8,
        # )

        # for dx, dy in self.get_shape():
        #     x, y = tetro_x + int(dx + self.offset), tetro_y + int(dy + self.offset)
        #     rel_x, rel_y = x + view_radius_x, y + view_radius_y

        #     if 0 <= rel_x < 2 * view_radius_x + 1 and 0 <= rel_y < 2 * view_radius_y + 1:
        #         view[rel_x, rel_y] = self.value

        tetro_x, tetro_y = self.position
        view = np.full(
            shape=(3, 3),
            fill_value=0,
            dtype=int,
        )

        # Populate view matrix with tetromino blocks
        for dx, dy in self.get_shape():
            # Calculate absolute and relative coordinates
            rel_x = tetro_x + int(dx + self.offset) + 1
            rel_y = tetro_y + int(dy + self.offset) + 1

            # Check if the block is within view bounds
            if 0 <= rel_x <= 2 and 0 <= rel_y <= 2:
                view[rel_x, rel_y] = 1

        return tuple(view.flatten())
    
    #endregion

    def build_agent(self, dir: Path = None, verbose: bool = False) -> Agent:
        """Build an agent for the Tetris environment.
        
        Args:
            dir (Path, optional): Directory to load prediction trees from. Defaults to None.
            verbose (bool, optional): Verbosity flag. Defaults to False.
        
        Returns:
            Agent: Agent object for the Tetris environment.
        """
        agent = Agent(
            {
                action: self.__getattribute__(action_name)
                for action, action_name in TETRIS_ACTIONS.items()
            },
            {
                perception: self.__getattribute__(perception_name)
                for perception, perception_name in TETRIS_PERCEPTIONS.items()
            }
        )

        if dir:
            if verbose:
                print(f"Loaded prediction trees from {dir}")
            # Load prediction trees from the specified directory
            agent.load(dir, verbose=verbose)
            
        if verbose:
            print(f"Agent created with {len(agent.action_dict)} actions and {len(agent.perception_dict)} perceptions.")

        return agent

    def get_state(self) -> dict:
        """Returns the current tetromino state as a dictionary.
        
        Returns:
            dict: Dictionary representation of the Tetromino state        
        """
        return {
            "name": self.name,
            "position": self.position,
            "orientation": self.orientation,
        }

    def from_state(self, state: dict):
        """Creates a new tetromino from state.

        Args:
            state (dict): Dictionary representation of the Tetromino state
        """
        return self.__init__(**state)