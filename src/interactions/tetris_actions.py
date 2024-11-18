from ..environment import TetrisEnv

class TetrisActions():

    def __init__(self, env: TetrisEnv):
        self.env = env

    def move_left(self):
        current_pos = self.env.current_tetro.pos
        new_pos = current_pos[0] - 1, current_pos[1]
        
        if self.env.check_collision(new_pos=new_pos):
            self.env.current_tetro.pos = new_pos

    def move_right(self):
        current_pos = self.env.current_tetro.pos
        new_pos = current_pos[0] + 1, current_pos[1]
        
        if self.env.check_collision(new_pos=new_pos):
            self.env.current_tetro.pos = new_pos

    def move_down(self):
        current_pos = self.env.current_tetro.pos
        new_pos = current_pos[0], current_pos[1] + 1
        
        if self.env.check_collision(new_pos=new_pos):
            self.env.current_tetro.pos = new_pos

    def rotate_cw(self):
        current_orientation = self.env.current_tetro.orientation
        new_orientation = (current_orientation + 1) % 4 
        
        if self.env.check_collision(new_orientation=new_orientation):
            self.env.current_tetro.orientation = new_orientation