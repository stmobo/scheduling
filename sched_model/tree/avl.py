from __future__ import annotations

from typing import TypeVar

from .base import Tree, TreeNode

K = TypeVar("K")
V = TypeVar("V")


class AVLTree(Tree):
    def __init__(self):
        super().__init__(AVLNode)


class AVLNode(TreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._balance = 0

    def _rotate(self):
        parent: AVLNode[K, V] = self._parent
        new_parent_bal = parent._balance
        new_pivot_bal = self._balance

        if self._is_left_child():
            new_parent_bal += 1
            if self._balance < 0:
                new_parent_bal -= self._balance

            new_pivot_bal += 1
            if new_parent_bal > 0:
                new_pivot_bal += new_parent_bal
        else:
            new_parent_bal -= 1
            if self._balance > 0:
                new_parent_bal -= self._balance

            new_pivot_bal -= 1
            if new_parent_bal < 0:
                new_pivot_bal += new_parent_bal

        self._balance = new_pivot_bal
        parent._balance = new_parent_bal

        super()._rotate()

    def _rebalance(self) -> AVLNode[K, V]:
        if self._balance < 0:
            child = self._left
        else:
            child = self._right

        if self._balance < 0 and child._balance > 0:
            child = child._right
            child._rotate()
        elif self._balance > 0 and child._balance < 0:
            child = child._left
            child._rotate()

        child._rotate()
        return child

    def _repair_insert(self):
        if self._parent is None:
            return

        parent: AVLNode[K, V] = self._parent
        old_bal = parent._balance

        if self._is_left_child():
            parent._balance -= 1
            if old_bal < 0:
                parent._rebalance()
                return
            elif old_bal > 0:
                return
        else:
            parent._balance += 1
            if old_bal > 0:
                parent._rebalance()
                return
            elif old_bal < 0:
                return

        return parent._repair_insert()

    def _repair_delete(self):
        parent: AVLNode[K, V] = self._parent
        if parent is None:
            return

        rebalance_required: bool = False
        sibling: AVLNode[K, V] = None

        if self._is_left_child():
            parent._balance += 1
            rebalance_required = parent._balance == 2
            sibling = parent._right
        else:
            parent._balance -= 1
            rebalance_required = parent._balance == -2
            sibling = parent._left

        if parent._balance == 0:
            # Height decrease was not absorbed at parent, but no rotations are
            # required here. Continue retracing:
            return parent._repair_delete()
        elif rebalance_required:
            sibling_bal = sibling._balance
            pivot = parent._rebalance()
            if sibling_bal != 0:
                # Height decrease was not absorbed at parent, continue retracing:
                return pivot._repair_delete()

        # Otherwise, the previous parent balance was previously zero, and we
        # can stop retracing here.

    def _print_node(self) -> str:
        return "{}: {:2d}".format(self.key, self._balance)
