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
        tree: Tree[K, V],
        parent: Optional[TreeNode[K, V]],
        prev: Union[SentinelNode, TreeNode[K, V]],
        next: Union[SentinelNode, TreeNode[K, V]],
    ):
        self._key: K = key
        self.value: V = None

        self_cls = self.__class__

        self._parent: Optional[self_cls[K, V]] = parent
        self._left: Optional[self_cls[K, V]] = None
        self._right: Optional[self_cls[K, V]] = None
        self._prev: Union[None, SentinelNode, self_cls[K, V]] = prev
        self._next: Union[None, SentinelNode, self_cls[K, V]] = next
        self._tree: Tree[K, V] = tree

        prev._next = self
        next._prev = self

    @property
    def key(self) -> K:
        """The key associated with this node.

        This property is immutable.
        """
        return self._key

    @property
    def prev(self) -> Optional[TreeNode[K, V]]:
        """This node's predecessor in the tree, if any."""
        ret = self._prev
        if isinstance(ret, TreeNode):
            return ret

    @property
    def next(self) -> Optional[TreeNode[K, V]]:
        """This node's successor in the tree, if any."""
        ret = self._next
        if isinstance(ret, TreeNode):
            return ret

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
        self._key = other._key
        self.value = other.value

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

        self._tree = None
        self._parent = None
        self._left = None
        self._right = None
        self._prev = None
        self._next = None

    def _find_node(self, key: K) -> TreeNode[K, V]:
        if self.key == key:
            return self
        elif key < self.key:
            if self._left is not None:
                return self._left._find_node(key)
        else:
            if self._right is not None:
                return self._right._find_node(key)

        raise KeyError(key)

    def _insert_node(
        self,
        key: K,
        prev: Union[SentinelNode, TreeNode[K, V]],
        next: Union[SentinelNode, TreeNode[K, V]],
    ) -> Tuple[bool, TreeNode[K, V]]:
        if self.key == key:
            return (False, self)

        if key < self.key:
            if self._left is not None:
                return self._left._insert_node(key, prev, self)
            else:
                new_node = self.__class__(key, self._tree, self, prev, self)
                self._set_left_child(new_node)
        else:
            if self._right is not None:
                return self._right._insert_node(key, self, next)
            else:
                new_node = self.__class__(key, self._tree, self, self, next)
                self._set_right_child(new_node)

        new_node._repair_insert()
        return (True, new_node)

    def _delete_node(self):
        if self._left is not None and self._right is not None:
            self._copy_data(self._next)
            return self._next._delete_node()
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

    def _print_recursive(self, level: int) -> str:
        ret = ""
        if self._left is not None:
            ret = self._left._print_recursive(level + 1)

        ret += ("    " * level) + self._print_node() + "\n"

        if self._right is not None:
            ret += self._right._print_recursive(level + 1)

        return ret

    # methods for subclasses to override:

    def _print_node(self) -> str:
        return str(self.key)

    def _delete_single_child(self):
        if self._left is not None:
            self._copy_data(self._left)
            return self._left._delete_node()
        elif self._right is not None:
            self._copy_data(self._right)
            return self._right._delete_node()
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

    def get_node(self, key: K) -> TreeNode[K, V]:
        """Directly retrieve a node within this tree.
        
        Raises KeyError if the tree does not contain the given key.
        """
        if self._root is None:
            raise KeyError(key)
        return self._root._find_node(key)

    def get_or_insert_node(self, key: K) -> Tuple[bool, TreeNode[K, V]]:
        """Retrieve a node within this tree, inserting a new node if one does
        not exist for the given key.

        Returns a tuple containing:
            - Whether a new node was inserted or not
            - The (possibly newly-inserted) node for the given key
        """
        if self._root is None:
            self._root = self._node_cls(key, self, None, self._sentinel, self._sentinel)
            self._len = 1
            return (True, self._root)
        else:
            created_new, insert_node = self._root._insert_node(
                key, self._sentinel, self._sentinel
            )
            if created_new:
                self._len += 1
            return (created_new, insert_node)

    def insert(self, key: K, val: V) -> Optional[V]:
        _, insert_node = self.get_or_insert_node(key)
        old_val = insert_node.value
        insert_node.value = val
        return old_val

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
        return (node.key, node.value)

    def max(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._last_node()
        return (node.key, node.value)

    def pop_min(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._first_node()
        r = (node.key, node.value)
        node._delete_node()
        self._len -= 1
        return r

    def pop_max(self) -> Tuple[K, V]:
        # pylint: disable=no-member
        node = self._last_node()
        r = (node.key, node.value)
        node._delete_node()
        self._len -= 1
        return r

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if self._root is not None:
            try:
                removed = self._root._find_node(key)
                val = removed.value
                removed._delete_node()
                self._len -= 1
                return val
            except KeyError:
                pass

        if default is not None:
            return default
        raise KeyError(key)

    def upper_bound(self, bound: K) -> Optional[Tuple[K, V]]:
        if self._root is None:
            return None

        node = self._root._upper_bound(bound)

        if isinstance(node, SentinelNode):
            return None
        return (node.key, node.value)

    def lower_bound(self, bound: K) -> Optional[Tuple[K, V]]:
        if self._root is None:
            return None

        node = self._root._lower_bound(bound)

        if isinstance(node, SentinelNode):
            return None
        return (node.key, node.value)

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

    def nodes(
        self,
        start_node: TreeNode[K, V],
        end_node: TreeNode[K, V],
        reverse: bool = False,
    ) -> Iterator[TreeNode[K, V]]:
        return TreeIter(TreeIter.NODES, start_node, end_node, reverse)

    def print(self) -> str:
        if self._root is not None:
            return self._root._print_recursive(0)
        else:
            return "<empty tree>"

    def __getitem__(self, key: K) -> V:
        return self.get_node(key).value

    def __setitem__(self, key: K, val: V):
        self.insert(key, val)

    def __delitem__(self, key: K):
        self.get_node(key)._delete_node()
        self._len -= 1

    def __contains__(self, key: K) -> bool:
        try:
            self.get_node(key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[K]:
        return self.keys()

    def __reversed__(self) -> Iterator[K]:
        return self.keys(reverse=True)

    def __len__(self) -> int:
        return self._len
