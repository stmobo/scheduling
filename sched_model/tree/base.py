from __future__ import annotations

from collections.abc import MutableMapping
from typing import Generic, TypeVar, Optional, Iterator, Tuple, Union, Type

from .iter import SentinelNode, TreeIter

K = TypeVar("K")
V = TypeVar("V")


class TreeNode(Generic[K, V]):
    def __init__(
        self,
        key: K,
        val: V,
        tree: Tree[K, V],
        parent: Optional[TreeNode[K, V]],
        prev: Union[SentinelNode, TreeNode[K, V]],
        next: Union[SentinelNode, TreeNode[K, V]],
    ):
        self.key: K = key
        self.val: V = val

        self_cls = self.__class__

        self._parent: Optional[self_cls[K, V]] = parent
        self._left: Optional[self_cls[K, V]] = None
        self._right: Optional[self_cls[K, V]] = None
        self._prev: Union[SentinelNode, self_cls[K, V]] = prev
        self._next: Union[SentinelNode, self_cls[K, V]] = next
        self._tree: Tree[K, V] = tree

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

    def _sibling(self) -> TreeNode[K, V]:
        parent = self._parent
        if parent is None:
            return None
        elif parent._left == self:
            return parent._right
        else:
            return parent._left

    def _rotate(self):
        parent: TreeNode[K, V] = self._parent
        gp: Optional[TreeNode[K, V]] = parent._parent
        parent_was_left = parent._is_left_child()

        if self._is_left_child():
            # Right rotation:
            parent._set_left_child(self._right)
            self._set_right_child(parent)
        else:
            # Left rotation:
            parent._set_right_child(self._left)
            self._set_left_child(parent)

        if gp is not None:
            if parent_was_left:
                gp._set_left_child(self)
            else:
                gp._set_right_child(self)
        else:
            self._parent = None
            self._tree._root = self

    def _unlink(self, replace_with: Optional[TreeNode[K, V]] = None):
        self._prev._next = self._next
        self._next._prev = self._prev

        if self._parent is not None:
            if self._is_left_child():
                self._parent._set_left_child(replace_with)
            else:
                self._parent._set_right_child(replace_with)
        else:
            # this was the root node:
            if replace_with is not None:
                replace_with._parent = None
            self._tree._root = replace_with

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
                new_node = self.__class__(key, val, self._tree, self, prev, self)
                self._set_left_child(new_node)
        else:
            if self._right is not None:
                return self._right.insert(key, val, self, next)
            else:
                new_node = self.__class__(key, val, self._tree, self, self, next)
                self._set_right_child(new_node)

        new_node._repair_insert()
        return None

    def delete(self):
        if self._left is not None and self._right is not None:
            self._copy_data(self._next)
            return self._next.delete()
        return self._delete_single_child()

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

    def print_recursive(self, level: int) -> str:
        ret = ""
        if self._left is not None:
            ret = self._left.print_recursive(level + 1)

        ret += ("    " * level) + self._print_node() + "\n"

        if self._right is not None:
            ret += self._right.print_recursive(level + 1)

        return ret

    def _print_node(self) -> str:
        return str(self.key)

    def _delete_single_child(self):
        if self._left is not None:
            self._copy_data(self._left)
            return self._left.delete()
        elif self._right is not None:
            self._copy_data(self._right)
            return self._right.delete()
        else:
            self._repair_delete()
            self._unlink()

    def _repair_delete(self):
        pass

    def _repair_insert(self):
        pass


class Tree(Generic[K, V], MutableMapping):
    def __init__(self, node_class: Type[TreeNode] = TreeNode):
        self._node_cls = node_class
        self._root: Optional[TreeNode[K, V]] = None
        self._sentinel = SentinelNode()
        self._len: int = 0

    def insert(self, key: K, val: V) -> Optional[V]:
        if self._root is None:
            self._root = self._node_cls(
                key, val, self, None, self._sentinel, self._sentinel
            )
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
            return self._root.print_recursive(0)
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
