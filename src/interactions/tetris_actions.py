from ..environment import TetrisEnv
from .tetris_vision import TetrisVision

class TetrisActions:
    def __init__(self, env: TetrisEnv, vision: TetrisVision):
        """
        Handles actions that can be performed on the Tetris environment by an agent.
        """
        self.env = env
        self.vision = vision

    def move_tetromino(self, dx: int, dy: int):
        """
        Move the current Tetromino by (dx, dy) if no collision occurs.
        """
        current_pos = self.env.current_tetromino.pos
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)

        if not self.env.is_collision(new_pos=new_pos):
            self.env.current_tetromino.pos = new_pos

    def rotate_tetromino(self, clockwise: bool = True):
        """
        Rotate the current Tetromino clockwise or counterclockwise if no collision occurs.
        """
        current_orientation = self.env.current_tetromino.orientation
        new_orientation = (current_orientation + 1) % 4 if clockwise else (current_orientation - 1) % 4

        if not self.env.is_collision(new_orientation=new_orientation):
            self.env.current_tetromino.orientation = new_orientation

    def move_tetromino_left(self):
        """Move the current Tetromino one step to the left."""
        self.move_tetromino(dx=-1, dy=0)

    def move_tetromino_right(self):
        """Move the current Tetromino one step to the right."""
        self.move_tetromino(dx=1, dy=0)

    def move_tetromino_down(self):
        """Move the current Tetromino one step down."""
        self.move_tetromino(dx=0, dy=1)

    def rotate_tetromino_cw(self):
        """Rotate the current Tetromino clockwise."""
        self.rotate_tetromino(clockwise=True)

    def rotate_tetromino_ccw(self):
        """Rotate the current Tetromino counterclockwise."""
        self.rotate_tetromino(clockwise=False)

    def move_view(self, dx: int, dy: int):
        """
        Move the current viewing window by (dx, dy) if not ouside viewing limits.
        """
        current_view_pos = self.vision.view_pos
        new_view_pos = (current_view_pos[0] + dx, current_view_pos[1] + dy)

        if self.vision.is_inside_viewing_border(new_view_pos):
            self.vision.view_pos = new_view_pos

    def move_view_left(self):
        """Move the current viewing window one step to the left."""
        self.move_view(dx=-1, dy=0)

    def move_view_right(self):
        """Move the current viewing window one step to the right."""
        self.move_view(dx=1, dy=0)

    def move_view_up(self):
        """Move the current viewing window one step up."""
        self.move_view(dx=0, dy=-1)

    def move_view_down(self):
        """Move the current viewing window one step down."""
        self.move_view(dx=0, dy=1)
        

    # def hard_drop(self):
    #     """
    #     Perform a hard drop of the Tetromino to the lowest possible position.
    #     """
    #     while True:
    #         current_pos = self.env.current_tetromino.pos
    #         new_pos = (current_pos[0], current_pos[1] + 1)

    #         if self.env.is_collision(new_pos=new_pos):
    #             break
    #         self.env.current_tetromino.pos = new_pos
