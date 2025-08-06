import numpy as np
from pathlib import Path
import random
import csv

from predictris.agent import Agent


TETROMINO_NAMES = ["T", "J", "I", "O", "L", "Z", "S"]
TETRIS_ACTIONS = {
    0: "move_tetromino_up",
    1: "move_tetromino_left",
    2: "move_tetromino_right",
    3: "move_tetromino_down",
    4: "rotate_tetromino_cw",
}
TETRIS_PERCEPTIONS = {
    0: "vision",
}


# Load valid states and create lookup
VALID_STATES = {}
with open(Path(__file__).parent / 'valid_states.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['name']
        if name not in VALID_STATES:
            VALID_STATES[name] = []
        VALID_STATES[name].append((
            name,
            int(row['pos_x']),
            int(row['pos_y']),
            int(row['orientation']),
        ))

# Load observations and create lookup
STATE_TO_OBS = {}
with open(Path(__file__).parent / 'state_to_obs.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        state_key = (row['name'], 
                    int(row['pos_x']), 
                    int(row['pos_y']), 
                    int(row['orientation']))
        STATE_TO_OBS[state_key] = tuple(int(row[f'obs_{i}']) for i in range(9))

# Load valid actions and create lookup
VALID_ACTIONS = {}
with open(Path(__file__).parent / 'valid_actions.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        from_key = (row['from_name'], 
                   int(row['from_x']), 
                   int(row['from_y']), 
                   int(row['from_orientation']))
        if from_key not in VALID_ACTIONS:
            VALID_ACTIONS[from_key] = {}
        VALID_ACTIONS[from_key][int(row['action'])] = (
            row['to_name'],
            int(row['to_x']),
            int(row['to_y']),
            int(row['to_orientation'])
        )

class TetrisEnvironment:
    """Represents the Tetris game environment."""

    def __init__(self, state: dict = None):
        """Initialize the Tetris environment with a given Tetromino state."""
        if state is None:
            self.random_init()
        else:
            self.state = state
        
    def random_init(self, tetrominos: list[str] = TETROMINO_NAMES):
        """Create a random environment with a valid tetromino configuration."""
        name = random.choice(tetrominos)
        self.state = random.choice(VALID_STATES[name])
        
    def vision(self, agent: Agent) -> tuple:
        """Get precomputed observation for current state."""
        return STATE_TO_OBS[self.state]

    def act(self, action_id: int) -> bool:
        """Try to apply an action and update state if valid."""
        try:
            self.state = VALID_ACTIONS[self.state][action_id]
        except KeyError:
            pass # Invalid action, do nothing

    def build_agent(self, depth: int, dir: Path = None, metrics: list[str] | None = None, verbose: bool = False) -> Agent:
        """Build an agent for the Tetris environment."""
        agent = Agent(
            {
                0: lambda agent: self.act(0),  # Move up
                1: lambda agent: self.act(1),  # Move left
                2: lambda agent: self.act(2),  # Move right
                3: lambda agent: self.act(3),  # Move down
                4: lambda agent: self.act(4),  # Rotate clockwise
            },
            {
                0: self.vision,  # Single perception method
            },
            depth=depth,
            metrics=metrics,
        )

        if dir:
            agent.load(dir, verbose=verbose)            
            
        return agent