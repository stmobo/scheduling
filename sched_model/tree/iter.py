from __future__ import annotations

from typing import Union

from . import base


class SentinelNode(object):
    def __init__(self):
        self._prev = self
        self._next = self


class TreeIter(object):
    KEYS = 0
    VALS = 1
    ITEMS = 2
    NODES = 3

    def __init__(
        self,
        mode: int,
        lower: Union[SentinelNode, base.TreeNode],
        upper: Union[SentinelNode, base.TreeNode],
        rev: bool,
    ):
        self._rev: bool = rev
        self._mode: int = mode
        self._cur: Union[None, SentinelNode, base.TreeNode] = None
        self._end: Union[None, SentinelNode, base.TreeNode] = None

        if lower._prev != upper:
            if not rev:
                self._cur = lower
                self._end = upper
            else:
                self._cur = upper
                self._end = lower

    def __iter__(self) -> TreeIter:
        return self

    def __reversed__(self) -> TreeIter:
        return TreeIter(self._mode, self._end, self._cur, not self._rev)

    def __next__(self):
        if self._cur is None:
            raise StopIteration()

        cur_node = self._cur

        if self._cur != self._end:
            if not self._rev:
                self._cur = self._cur._next
            else:
                self._cur = self._cur._prev
        else:
            self._cur = None
            self._end = None

        if self._mode == TreeIter.KEYS:
            return cur_node.key
        elif self._mode == TreeIter.VALS:
            return cur_node.value
        elif self._mode == TreeIter.ITEMS:
            return (cur_node.key, cur_node.value)
        elif self._mode == TreeIter.NODES:
            return cur_node
