import numpy as np

class Agent():

    def __init__(
        self,
        actions: list[callable[[], None]],
        observe: callable[[], np.ndarray]
    ):
        self.actions = actions
        self.observe = observe

    