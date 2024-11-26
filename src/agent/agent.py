import numpy as np
from typing import Callable

class Agent():

    def __init__(
        self,
        actions: list[Callable[[], None]],
        observe: Callable[[], np.ndarray]
    ):
        self.actions = actions
        self.observe = observe

    