from __future__ import annotations

import numpy as np
from typing import Union


class Resources(object):
    def __init__(self, v: RscCompatible):
        self.resources: np.ndarray = self._resource_vec(v).astype(np.int)

    def clone(self) -> Resources:
        return Resources(self.resources)

    def valid(self) -> bool:
        return (self.resources >= 0).all()

    @staticmethod
    def _resource_vec(val: RscCompatible) -> np.ndarray:
        """Gets the `resources` vector out of a Resources object, and leaves
        other parameters untouched.
        """
        try:
            return val.resources
        except AttributeError:
            return val

    def all_geq(self, other: RscCompatible) -> bool:
        """Test whether all resources in this vector are greater than or 
        equal to all resources contained in `other`.
        """
        return (self.resources >= self._resource_vec(other)).all()

    def __add__(self, other: RscCompatible) -> Resources:
        return Resources(self.resources + self._resource_vec(other))

    def __sub__(self, other: RscCompatible) -> Resources:
        return Resources(self.resources - self._resource_vec(other))

    def __radd__(self, other: RscCompatible) -> Resources:
        return self.__add__(other)

    def __rsub__(self, other: RscCompatible) -> Resources:
        return Resources(self._resource_vec(other) - self.resources)

    def __iadd__(self, other: RscCompatible) -> Resources:
        self.resources += self._resource_vec(other)
        return self

    def __isub__(self, other: RscCompatible) -> Resources:
        self.resources -= self._resource_vec(other)
        return self

    def __iter__(self):
        return self.resources.__iter__()

    def __len__(self) -> int:
        return len(self.resources)

    def __str__(self) -> str:
        return str(self.resources)

    def __repr__(self) -> str:
        return "Resources(" + repr(self.resources) + ")"


RscCompatible = Union[Resources, np.ndarray]
