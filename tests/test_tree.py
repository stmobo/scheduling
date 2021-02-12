from hypothesis import given, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant
import pytest

from sched_model.tree.base import Tree
from sched_model.tree.rb import RBTree, RBNode
from sched_model.tree.avl import AVLTree, AVLNode


@st.composite
def dict_and_key(draw, keys=st.integers(), values=st.uuids()):
    d = draw(st.dictionaries(keys, values, min_size=1))
    key = draw(st.sampled_from(list(d.keys())))
    return (d, key)


@st.composite
def dict_and_subset(draw, keys=st.integers(), values=st.uuids()):
    d = draw(st.dictionaries(keys, values, min_size=1))
    subset = sorted(d.keys())

    i1 = draw(st.integers(min_value=0, max_value=len(subset) - 1))
    subset = subset[i1:]

    i2 = draw(st.integers(min_value=1, max_value=len(subset)))
    subset = subset[:i2]

    upper_bound = subset.pop()

    if len(subset) == 0:
        lower_bound = upper_bound
    else:
        lower_bound = subset[0]

    return (d, subset, lower_bound, upper_bound)


def verify_rb_integrity(cur: RBNode, seen_keys: dict) -> int:
    assert cur.key not in seen_keys, "encountered loop in tree pointers at node " + str(
        cur.key
    )
    seen_keys[cur.key] = cur.val

    if cur._left is not None:
        assert (not cur._red) or (
            not cur._left._red
        ), "Red node {} has red left child {}".format(str(cur.key), str(cur._left.key))

        assert (
            cur._left._parent == cur
        ), "parent <> left child link broken at node {} (child = {})".format(
            str(cur.key), str(cur._left.key)
        )

        left_blk_height = verify_rb_integrity(cur._left, seen_keys)
    else:
        left_blk_height = 1

    if cur._right is not None:
        assert (not cur._red) or (
            not cur._right._red
        ), "Red node {} has red right child {}".format(
            str(cur.key), str(cur._right.key)
        )

        assert (
            cur._right._parent == cur
        ), "parent <> right child link broken at node {} (child = {})".format(
            str(cur.key), str(cur._right.key)
        )
        right_blk_height = verify_rb_integrity(cur._right, seen_keys)
    else:
        right_blk_height = 1

    # Verify balance constraint
    assert (
        left_blk_height == right_blk_height
    ), "Left and right subtrees have different black heights ({} != {})".format(
        left_blk_height, right_blk_height
    )

    # Return black height of this subtree
    if cur._red:
        return left_blk_height
    else:
        return left_blk_height + 1


def verify_avl_integrity(cur: AVLNode, seen_keys: dict) -> int:
    assert cur.key not in seen_keys, "encountered loop in tree pointers at node " + str(
        cur.key
    )
    seen_keys[cur.key] = cur.val

    if cur._left is not None:
        assert (
            cur._left._parent == cur
        ), "parent <> left child link broken at node {} (child = {})".format(
            str(cur.key), str(cur._left.key)
        )
        left_height = verify_avl_integrity(cur._left, seen_keys)
    else:
        left_height = 0

    if cur._right is not None:
        assert (
            cur._right._parent == cur
        ), "parent <> right child link broken at node {} (child = {})".format(
            str(cur.key), str(cur._right.key)
        )
        right_height = verify_avl_integrity(cur._right, seen_keys)
    else:
        right_height = 0

    # Verify balance constraint
    assert (
        abs(right_height - left_height) <= 1
    ), "balance constraint violated at node " + str(cur.key)

    # Return height of this subtree
    return max(right_height, left_height) + 1


def verify_tree_integrity(tree: AVLTree, items: dict):
    seen_keys = {}
    if tree._root is not None:
        if isinstance(tree._root, AVLNode):
            verify_avl_integrity(tree._root, seen_keys)
        elif isinstance(tree._root, RBNode):
            assert not tree._root._red, "RBTree root is not black"
            verify_rb_integrity(tree._root, seen_keys)

    assert len(seen_keys) == len(
        items
    ), "tree traversal returned incorrect number of items (got {}, expected {})".format(
        len(seen_keys), len(items)
    )

    assert len(tree) == len(
        seen_keys
    ), "tree stored length differs from traversed number of nodes (got {}, expected {})".format(
        len(tree), len(seen_keys)
    )

    for k, v in items.items():
        assert (
            seen_keys[k] == v
        ), "found incorrect key-value mapping during tree traversal (key {}: got {}, expected {})".format(
            str(k), str(seen_keys[k]), str(v)
        )


class TreeStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.tree = Tree()
        self.model = {}

    keys = Bundle("keys")
    values = Bundle("values")

    @invariant()
    def check_integrity(self):
        verify_tree_integrity(self.tree, self.model)

    @rule(target=keys, k=st.integers())
    def add_key(self, k):
        return k

    @rule(target=values, v=st.uuids())
    def add_value(self, v):
        return v

    @rule(k=keys, v=values)
    def setitem(self, k, v):
        self.tree[k] = v
        self.model[k] = v

    @rule(k=keys)
    def getitem(self, k):
        if k not in self.model:
            with pytest.raises(KeyError):
                self.tree[k]
        else:
            assert self.tree[k] == self.model[k]

    @rule(k=keys)
    def delitem(self, k):
        if k not in self.model:
            with pytest.raises(KeyError):
                del self.tree[k]
        else:
            del self.tree[k]
            del self.model[k]

    @rule(k=keys)
    def contains(self, k):
        assert (k in self.tree) == (k in self.model)

    @rule(k=keys, v=values)
    def insert_method(self, k, v):
        assert self.tree.insert(k, v) == self.model.get(k)
        self.model[k] = v

    @rule(k=keys)
    def pop(self, k):
        if k not in self.model:
            with pytest.raises(KeyError):
                self.tree.pop(k)
        else:
            assert self.tree.pop(k) == self.model.pop(k)

    @rule()
    def min(self):
        if len(self.model) == 0:
            with pytest.raises(IndexError):
                self.tree.min()
        else:
            s = sorted(self.model.keys())
            k = s[0]
            v = self.model[k]
            assert self.tree.min() == (k, v)

    @rule()
    def max(self):
        if len(self.model) == 0:
            with pytest.raises(IndexError):
                self.tree.max()
        else:
            s = sorted(self.model.keys())
            k = s[-1]
            v = self.model[k]
            assert self.tree.max() == (k, v)

    @rule()
    def pop_min(self):
        if len(self.model) == 0:
            with pytest.raises(IndexError):
                self.tree.pop_min()
        else:
            s = sorted(self.model.keys())
            k = s[0]
            v = self.model[k]
            assert self.tree.pop_min() == (k, v)
            del self.model[k]

    @rule()
    def pop_max(self):
        if len(self.model) == 0:
            with pytest.raises(IndexError):
                self.tree.pop_max()
        else:
            s = sorted(self.model.keys())
            k = s[-1]
            v = self.model[k]
            assert self.tree.pop_max() == (k, v)
            del self.model[k]

    @rule(k=keys)
    def get(self, k):
        assert self.tree.get(k) == self.model.get(k)


class AVLTreeStateMachine(TreeStateMachine):
    def __init__(self):
        super().__init__()
        self.tree = AVLTree()


class RBTreeStateMachine(TreeStateMachine):
    def __init__(self):
        super().__init__()
        self.tree = RBTree()


TestAVLTreeStateMachine = AVLTreeStateMachine.TestCase
TestRBTreeStateMachine = RBTreeStateMachine.TestCase


@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_insert_positive(tree_type, givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    verify_tree_integrity(tree, items)
    assert tree[test_key] == items[test_key]


@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_delete(tree_type, givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        tree[k] = v

    del tree[test_key]
    del items[test_key]
    verify_tree_integrity(tree, items)


@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_nonexistent_key(tree_type, givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        tree[k] = v
    del tree[test_key]

    try:
        _blah = tree[test_key]
        assert False
    except KeyError:
        pass

    try:
        del tree[test_key]
        assert False
    except KeyError:
        pass

    assert test_key not in tree


@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_pop(tree_type, givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        tree[k] = v

    assert tree.pop(test_key) == items.pop(test_key)


@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
@given(st.dictionaries(st.integers(), st.uuids()))
def test_iter(tree_type, items):
    keys = sorted(items.keys())
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    ret = list(tree.items())

    assert len(ret) == len(keys)
    for k1, kv in zip(keys, ret):
        assert k1 == kv[0]
        assert items[k1] == kv[1]


@given(st.dictionaries(st.integers(), st.uuids()))
@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
def test_iter_reverse(tree_type, items):
    keys = sorted(items.keys(), reverse=True)
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    ret = list(tree.items(reverse=True))

    assert len(ret) == len(keys)
    for k1, kv in zip(keys, ret):
        assert k1 == kv[0]
        assert items[k1] == kv[1]


@given(dict_and_subset())  # pylint: disable=no-value-for-parameter
@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
def test_iter_bounds(tree_type, givens):
    items, subset, lower_bound, upper_bound = givens
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    ret = list(tree.items(left_bound=lower_bound, right_bound=upper_bound))

    assert len(ret) == len(subset)
    for k1, kv in zip(subset, ret):
        assert k1 == kv[0]
        assert items[k1] == kv[1]


@given(dict_and_subset())  # pylint: disable=no-value-for-parameter
@pytest.mark.parametrize("tree_type", [AVLTree, RBTree])
def test_iter_bounds_reverse(tree_type, givens):
    items, subset, lower_bound, upper_bound = givens
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    ret = list(
        tree.items(left_bound=lower_bound, right_bound=upper_bound, reverse=True)
    )

    assert len(ret) == len(subset)
    for k1, kv in zip(reversed(subset), ret):
        assert k1 == kv[0]
        assert items[k1] == kv[1]
