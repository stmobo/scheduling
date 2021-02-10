from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar, Optional, Iterator, Tuple, Union

K = TypeVar("K")
V = TypeVar("V")


class AVLTree(Generic[K, V], MutableMapping):
    def __init__(self):
        self._root: Optional[TreeNode[K, V]] = None
        self._sentinel = SentinelNode()
        self._len: int = 0

    def insert(self, key: K, val: V) -> Optional[V]:
        if self._root is None:
            self._root = TreeNode(key, val, self, self._sentinel, self._sentinel)
            self._len = 1
            return None
        else:
            ret = self._root.insert(key, val, self._sentinel, self._sentinel)
            if ret is None:
                self._len += 1
            return ret

    def _first_node(self) -> TreeNode[K, V]:
        node = self._sentinel._next
        if node == self._sentinel:
            raise IndexError("Tree is empty")
        return node

    def _last_node(self) -> TreeNode[K, V]:
        node = self._sentinel._prev
        if node == self._sentinel:
            raise IndexError("Tree is empty")
        return node

    def min(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._first_node()
        return (node.key, node.val)

    def max(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._last_node()
        return (node.key, node.val)

    def pop_min(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._first_node()
        r = (node.key, node.val)
        node.delete()
        self._len -= 1
        return r

    def pop_max(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._last_node()
        r = (node.key, node.val)
        node.delete()
        self._len -= 1
        return r

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if self._root is not None:
            try:
                removed = self._root.get(key)
                val = removed.val
                removed.delete()
                self._len -= 1
                return val
            except KeyError:
                pass

        if default is not None:
            return default
        raise KeyError(key)

    def _do_iter(
        self,
        mode: int,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> TreeIter:
        if (
            left_bound is not None
            and right_bound is not None
            and (left_bound > right_bound)
        ):
            return self._do_iter(mode, right_bound, left_bound, reverse)

        if left_bound is not None and self._root is not None:
            lb = self._root._lower_bound(left_bound)
        else:
            lb = self._sentinel._next

        if right_bound is not None and self._root is not None:
            rb = self._root._upper_bound(right_bound)
        else:
            rb = self._sentinel._prev

        return TreeIter(mode, lb, rb, reverse)

    def items(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[Tuple[K, V]]:
        return self._do_iter(TreeIter.ITEMS, left_bound, right_bound, reverse)

    def keys(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[K]:
        return self._do_iter(TreeIter.KEYS, left_bound, right_bound, reverse)

    def values(
        self,
        left_bound: Optional[K] = None,
        right_bound: Optional[K] = None,
        reverse: bool = False,
    ) -> Iterator[V]:
        return self._do_iter(TreeIter.VALS, left_bound, right_bound, reverse)

    def print(self) -> str:
        if self._root is not None:
            return self._root.print(0)
        else:
            return "<empty tree>"

    def __getitem__(self, key: K) -> V:
        if self._root is None:
            raise KeyError(key)
        return self._root.get(key).val

    def __setitem__(self, key: K, val: V):
        self.insert(key, val)

    def __delitem__(self, key: K):
        if self._root is None:
            raise KeyError(key)
        self._root.get(key).delete()
        self._len -= 1

    def __contains__(self, key: K) -> bool:
        if self._root is None:
            return False

        try:
            self._root.get(key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[K]:
        return self.keys()

    def __reversed__(self) -> Iterator[K]:
        return self.keys(reverse=True)

    def __len__(self) -> int:
        return self._len


class TreeNode(Generic[K, V]):
    def __init__(
        self,
        key: K,
        val: V,
        tree: AVLTree[K, V],
        prev: Union[SentinelNode, TreeNode[K, V]],
        next: Union[SentinelNode, TreeNode[K, V]],
    ):
        self.key: K = key
        self.val: V = val

        self._balance = 0
        self._parent: Optional[TreeNode[K, V]] = None
        self._left: Optional[TreeNode[K, V]] = None
        self._right: Optional[TreeNode[K, V]] = None
        self._prev: Union[SentinelNode, TreeNode[K, V]] = prev
        self._next: Union[SentinelNode, TreeNode[K, V]] = next
        self._tree: AVLTree[K, V] = tree

        prev._next = self
        next._prev = self

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

    def _copy_data(self, other: TreeNode[K, V]):
        self.key = other.key
        self.val = other.val

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

    def _retrace_delete(self):
        parent: TreeNode[K, V] = self._parent
        if parent is None:
            return

        rebalance_required: bool = False
        sibling: TreeNode[K, V] = None

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
            return parent._retrace_delete()
        elif rebalance_required:
            sibling_bal = sibling._balance
            pivot = parent._rebalance()
            if sibling_bal != 0:
                # Height decrease was not absorbed at parent, continue retracing:
                return pivot._retrace_delete()

        # Otherwise, the previous parent balance was previously zero, and we
        # can stop retracing here.

    def get(self, key: K) -> TreeNode[K, V]:
        if self.key == key:
            return self
        elif key < self.key:
            if self._left is not None:
                return self._left.get(key)
        else:
            if self._right is not None:
                return self._right.get(key)

        raise KeyError(key)

    def insert(
        self,
        key: K,
        val: V,
        prev: Union[SentinelNode, TreeNode[K, V]],
        next: Union[SentinelNode, TreeNode[K, V]],
    ) -> Optional[V]:
        if self.key == key:
            old_val = self.val
            self.val = val
            return old_val

        if key < self.key:
            if self._left is not None:
                return self._left.insert(key, val, prev, self)
            else:
                new_node = TreeNode(key, val, self._tree, prev, self)
                self._set_left_child(new_node)
        else:
            if self._right is not None:
                return self._right.insert(key, val, self, next)
            else:
                new_node = TreeNode(key, val, self._tree, self, next)
                self._set_right_child(new_node)

        new_node._retrace_insert()
        return None

    def delete(self):
        if self._left is not None and self._right is not None:
            self._copy_data(self._next)
            return self._next.delete()
        elif self._left is not None:
            self._copy_data(self._left)
            return self._left.delete()
        elif self._right is not None:
            self._copy_data(self._right)
            return self._right.delete()
        else:
            self._retrace_delete()

            self._prev._next = self._next
            self._next._prev = self._prev

            if self._parent is not None:
                if self._is_left_child():
                    self._parent._left = None
                else:
                    self._parent._right = None
            else:
                # this is the root node:
                self._tree._root = None

    # inclusive lower bound
    def _lower_bound(self, bound: K) -> Union[TreeNode[K, V], SentinelNode]:
        if bound == self.key:
            return self
        elif bound < self.key:
            if self._left is not None:
                return self._left._lower_bound(bound)
            else:
                return self
        else:
            if self._right is not None:
                return self._right._lower_bound(bound)
            else:
                return self._next

    # exclusive upper bound
    def _upper_bound(self, bound: K) -> Union[TreeNode[K, V], SentinelNode]:
        if bound <= self.key:
            if self._left is not None:
                return self._left._upper_bound(bound)
            else:
                return self._prev
        else:
            if self._right is not None:
                return self._right._upper_bound(bound)
            else:
                return self

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


class SentinelNode(object):
    def __init__(self):
        self._prev = self
        self._next = self


class TreeIter(object):
    KEYS = 0
    VALS = 1
    ITEMS = 2

    def __init__(
        self,
        mode: int,
        lower: Union[SentinelNode, TreeNode],
        upper: Union[SentinelNode, TreeNode],
        rev: bool,
    ):
        self._rev: bool = rev
        self._mode: int = mode
        self._cur: Union[None, SentinelNode, TreeNode] = None
        self._end: Union[None, SentinelNode, TreeNode] = None

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
            return cur_node.val
        else:
            return (cur_node.key, cur_node.val)
