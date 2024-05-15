from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any
import inspect

from synthetic_languages.process.Process import Process

# TODO: Write test for PROCESS_REGISTRY
# TODO: Think if you really need PROCESS_REGSITRY (if only getting called during dataloader creation, it may be better to have the dataloader take in a process)
# TODO: Add test to make sure that all members of this module are a member of Process
# TODO: Find paper where mess3 process is introduced
# TODO: Think through whether self.name is necessary (review it's usage in derive_mixed_state_presentation)
# TODO: Move _create_hmm into the init function prior to super()__init__()

class ZeroOneR(Process):
    name = "z1r"
    def __init__(self, prob_of_zero_from_r_state: float = 0.5):
        self.p = prob_of_zero_from_r_state
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        state_names = {"0": 0, "1": 1, "R": 2}
        T[0, state_names["0"], state_names["1"]] = 1.0
        T[1, state_names["1"], state_names["R"]] = 1.0
        T[0, state_names["R"], state_names["0"]] = self.p
        T[1, state_names["R"], state_names["0"]] = 1 - self.p

        return T, state_names


class RRXOR(Process):
    name = "rrxor"
    def __init__(self, pR1=0.5, pR2=0.5):
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 5, 5))
        state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
        T[0, state_names["S"], state_names["0"]] = self.pR1
        T[1, state_names["S"], state_names["1"]] = 1 - self.pR1
        T[0, state_names["0"], state_names["F"]] = self.pR2
        T[1, state_names["0"], state_names["T"]] = 1 - self.pR2
        T[0, state_names["1"], state_names["T"]] = self.pR2
        T[1, state_names["1"], state_names["F"]] = 1 - self.pR2
        T[1, state_names["T"], state_names["S"]] = 1.0
        T[0, state_names["F"], state_names["S"]] = 1.0

        return T, state_names


class Mess3(Process):
    name = "mess3"
    def __init__(self, x=0.15, a=0.6):
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names

class TransitionMatrixProcess(Process):
    name = "transition_matrix" # General purpose process for your favorite transition matrix
    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix
        super().__init__()

    def _create_hmm(self):
        return self.transition_matrix, {i: i for i in range(self.transition_matrix.shape[0])}

def create_process_registry(by_name: bool = False, items: Optional[Dict[str, Any]] = None) -> Dict[str, type]:
    items = items if items is not None else inspect.currentframe().f_globals # Globals is a way to get access to outside the func AFAIK
    d =  {key: value for key, value in items.items() if isinstance(value, type) and issubclass(value, Process) and key != 'Process'}
    if by_name:
        assert all([hasattr(value, "name") for value in d.values()]), "All processes must have a name attribute"
        assert len(set([value.name for value in d.values()])) == len(d.values()), "All processes must have a unique name"
        return {value.name: value for value in d.values()}
    return d
# Pass the current frame locals so we are not getting some giga globals if available
PROCESS_REGISTRY: Dict[str, type] = create_process_registry(by_name=True, items=inspect.currentframe().f_locals)
