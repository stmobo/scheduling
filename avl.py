from __future__ import annotations

from typing import Any, Generic, TypeVar, Optional, Iterator, Tuple

K = TypeVar("K")
V = TypeVar("V")


class AVLTree(Generic[K, V]):
    def __init__(self):
        self._root: Optional[TreeNode[K, V]] = None
        self._len: int = 0

    def print(self) -> str:
        if self._root is not None:
            return self._root.print(0)
        else:
            return "<empty tree>"

    def insert(self, key: K, val: V) -> Optional[V]:
        if self._root is None:
            self._root = TreeNode(key, val, self)
            self._len = 1
            return None
        else:
            ret = self._root.insert(key, val)
            if ret is None:
                self._len += 1
            return ret

    def get(self, key: K) -> V:
        if self._root is None:
            raise KeyError(key)
        return self._root.get(key)

    def _do_iter(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Optional[Iterator[TreeNode[K, V]]]:
        if self._root is None:
            return None
        elif reverse:
            return self._root.iter_reverse(left_bound, right_bound)
        else:
            return self._root.iter(left_bound, right_bound)

    def items(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[Tuple[K, V]]:
        it = self._do_iter(left_bound, right_bound, reverse)
        if it is None:
            return

        for node in it:
            yield (node.key, node.val)

    def keys(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[K]:
        it = self._do_iter(left_bound, right_bound, reverse)
        if it is None:
            return

        for node in it:
            yield node.key

    def values(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[V]:
        it = self._do_iter(left_bound, right_bound, reverse)
        if it is None:
            return

        for node in it:
            yield node.val

    def __iter__(self) -> Iterator[K]:
        return self.keys()

    def __reversed__(self) -> Iterator[K]:
        return self.keys(reverse=True)

    def __len__(self) -> int:
        return self._len


class TreeNode(Generic[K, V]):
    def __init__(self, key: K, val: V, tree: AVLTree[K, V]):
        self.key: K = key
        self.val: V = val

        self._balance = 0
        self._parent: Optional[TreeNode[K, V]] = None
        self._left: Optional[TreeNode[K, V]] = None
        self._right: Optional[TreeNode[K, V]] = None
        self._tree: AVLTree[K, V] = tree

    def _set_left_child(self, child: Optional[TreeNode[K, V]]):
        self._left = child
        if child is not None:
            child._parent = self

    def _set_right_child(self, child: Optional[TreeNode[K, V]]):
        self._right = child
        if child is not None:
            child._parent = self

    def _is_left_child(self) -> bool:
        return (self._parent is not None) and (self._parent._left == self)

    def _is_right_child(self) -> bool:
        return (self._parent is not None) and (self._parent._right == self)

    def _rotate(self):
        parent: TreeNode[K, V] = self._parent
        gp: Optional[TreeNode[K, V]] = parent._parent

        new_parent_bal = parent._balance
        new_pivot_bal = self._balance
        parent_was_left = parent._is_left_child()

        if self._is_left_child():
            # Right rotation:
            parent._set_left_child(self._right)
            self._set_right_child(parent)

            new_parent_bal += 1
            if self._balance < 0:
                new_parent_bal -= self._balance

            new_pivot_bal += 1
            if new_parent_bal > 0:
                new_pivot_bal += new_parent_bal
        else:
            # Left rotation:
            parent._set_right_child(self._left)
            self._set_left_child(parent)

            new_parent_bal -= 1
            if self._balance > 0:
                new_parent_bal -= self._balance

            new_pivot_bal -= 1
            if new_parent_bal < 0:
                new_pivot_bal += new_parent_bal

        self._balance = new_pivot_bal
        parent._balance = new_parent_bal

        if gp is not None:
            if parent_was_left:
                gp._set_left_child(self)
            else:
                gp._set_right_child(self)
        else:
            self._parent = None
            self._tree._root = self

    def _rebalance(self) -> TreeNode[K, V]:
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

    def _retrace_insert(self):
        if self._parent is None:
            return

        parent: TreeNode[K, V] = self._parent
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

        return parent._retrace_insert()

    def insert(self, key: K, val: V) -> Optional[V]:
        if self.key == key:
            old_val = self.val
            self.val = val
            return old_val

        if key < self.key:
            if self._left is not None:
                return self._left.insert(key, val)
            else:
                new_node = TreeNode(key, val, self._tree)
                self._set_left_child(new_node)
        else:
            if self._right is not None:
                return self._right.insert(key, val)
            else:
                new_node = TreeNode(key, val, self._tree)
                self._set_right_child(new_node)

        new_node._retrace_insert()
        return None

    def get(self, key: K) -> V:
        if self.key == key:
            return self.val
        elif key < self.key:
            if self._left is not None:
                return self._left.get(key)
        else:
            if self._right is not None:
                return self._right.get(key)

        raise KeyError(key)

    def iter(
        self, left_bound: Optional[K] = None, right_bound: Optional[K] = None
    ) -> Iterator[TreeNode[K, V]]:
        left_in_range = (
            (left_bound is None) or (self.key == left_bound) or (left_bound < self.key)
        )
        right_in_range = (right_bound is None) or (self.key < right_bound)

        if left_in_range and self._left is not None:
            yield from self._left.iter(left_bound, right_bound)

        if left_in_range and right_in_range:
            yield self

        if right_in_range and self._right is not None:
            yield from self._right.iter(left_bound, right_bound)

    def iter_reverse(
        self, left_bound: Optional[K] = None, right_bound: Optional[K] = None
    ) -> Iterator[TreeNode[K, V]]:
        left_in_range = (
            (left_bound is None) or (self.key == left_bound) or (left_bound < self.key)
        )
        right_in_range = (right_bound is None) or (self.key < right_bound)

        if right_in_range and self._right is not None:
            yield from self._right.iter_reverse(left_bound, right_bound)

        if left_in_range and right_in_range:
            yield self

        if left_in_range and self._left is not None:
            yield from self._left.iter_reverse(left_bound, right_bound)

    def print(self, level: int) -> str:
        ret = ""
        if self._left is not None:
            ret = self._left.print(level + 1)

        ret += ("    " * level) + "{}: {:2d}\n".format(self.key, self._balance)

        if self._right is not None:
            ret += self._right.print(level + 1)

        return ret
