from __future__ import annotations

from typing import TypeVar, Optional

from .base import Tree, TreeNode

K = TypeVar("K")
V = TypeVar("V")


class RBTree(Tree):
    def __init__(self):
        super().__init__(RBNode)


class RBNode(TreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._red = self._parent is not None

    def _repair_insert(self):
        if self._parent is None:
            self._red = False
            return

        parent: RBNode[K, V] = self._parent
        if not parent._red:
            return

        uncle: RBNode[K, V] = parent._sibling()
        grandparent: RBNode[K, V] = parent._parent
        if uncle is not None and uncle._red:
            parent._red = False
            uncle._red = False
            grandparent._red = True
            return grandparent._repair_insert()

        if self._is_left_child() != parent._is_left_child():
            self._rotate()
            parent = self

        parent._rotate()
        parent._red = False
        grandparent._red = True

    def _delete_single_child(self):
        replace_with = None
        if not self._red:
            if self._left is not None:
                replace_with = self._left
                self._left._red = False
            elif self._right is not None:
                replace_with = self._right
                self._right._red = False
            else:
                self._repair_delete()
        self._unlink(replace_with)

    def _repair_delete(self):
        if self._parent is None:
            return

        sibling: Optional[RBNode[K, V]] = self._sibling()
        parent: RBNode[K, V] = self._parent

        if sibling is not None and sibling._red:
            parent._red = True
            sibling._red = False
            sibling._rotate()
            sibling = self._sibling()

        sib_left_black = (sibling._left is None) or not sibling._left._red
        sib_right_black = (sibling._right is None) or not sibling._right._red

        if (
            (not parent._red)
            and (not sibling._red)
            and sib_left_black
            and sib_right_black
        ):
            sibling._red = True
            return parent._repair_delete()

        if parent._red and (not sibling._red) and sib_left_black and sib_right_black:
            sibling._red = True
            parent._red = False
            return

        if not sibling._red:
            if self._is_left_child() and (not sib_left_black) and sib_right_black:
                sibling._red = True
                sibling._left._red = False
                sibling._left._rotate()
                sibling = self._sibling()
            elif (
                (not self._is_left_child()) and sib_left_black and (not sib_right_black)
            ):
                sibling._red = True
                sibling._right._red = False
                sibling._right._rotate()
                sibling = self._sibling()

        sibling._red = parent._red
        parent._red = False

        if self._is_left_child():
            sibling._right._red = False
        else:
            sibling._left._red = False
        sibling._rotate()

    def _print_node(self) -> str:
        if self._red:
            return str(self.key) + " (R)"
        else:
            return str(self.key) + " (B)"
