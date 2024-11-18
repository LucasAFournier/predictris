from ..environment.tetris_env import TetrisEnv
import numpy as np


class Vision():
    
    def __init__(self, env: TetrisEnv, view_radius=1):
        
        self.env = env
        self.view_center = (env.grid.shape[0] // 2, env.grid.shape[1] // 2)
        
        self.view_shape = (2 * view_radius + 1, 2 * view_radius + 1)
        self.current_view = np.zeros(shape=self.view_shape)
        

    def observe(self):
        tetro = self.env.current_tetro
        tetro_pos = tetro.pos
        
        threshold = 2 + self.view_radius
        x_center_diff = tetro_pos[0] - self.view_center[0]
        y_center_diff = tetro_pos[1] - self.view_center[1]

        if (
            abs(x_center_diff) < threshold and
            abs(y_center_diff) < threshold
        ):
            tetro_shape = tetro.get_shape()
            tetro_value = tetro.value
            
            for coo_rel_tetro in tetro_shape:
                x_rel_tetro, y_rel_tetro = coo_rel_tetro
                x_rel_view = x_rel_tetro + x_center_diff
                y_rel_view = y_rel_tetro + y_center_diff
                if (
                    0 <= x_rel_view < self.view_shape[0] and
                    0 <= y_rel_view < self.view_shape[1]
                ):
                    self.current_view[x_rel_view, y_rel_view] = tetro_value
        
        return self.current_view