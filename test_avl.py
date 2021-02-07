from hypothesis import given, note, assume, strategies as st
from avl import AVLTree, TreeNode


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


def verify_node_integrity(cur: TreeNode, seen_keys: dict) -> int:
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
        left_height = verify_node_integrity(cur._left, seen_keys)
    else:
        left_height = 0

    if cur._right is not None:
        assert (
            cur._right._parent == cur
        ), "parent <> right child link broken at node {} (child = {})".format(
            str(cur.key), str(cur._right.key)
        )
        right_height = verify_node_integrity(cur._right, seen_keys)
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
        verify_node_integrity(tree._root, seen_keys)

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


@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_insert_positive(givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    verify_tree_integrity(tree, items)
    assert tree[test_key] == items[test_key]


@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_delete(givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        tree[k] = v

    del tree[test_key]
    del items[test_key]
    verify_tree_integrity(tree, items)


@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_nonexistent_key(givens):
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


@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_pop(givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        tree[k] = v

    assert tree.pop(test_key) == items.pop(test_key)


@given(st.dictionaries(st.integers(), st.uuids()))
def test_iter(items):
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
def test_iter_reverse(items):
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
def test_iter_bounds(g):
    items, subset, lower_bound, upper_bound = g
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    ret = list(tree.items(left_bound=lower_bound, right_bound=upper_bound))

    assert len(ret) == len(subset)
    for k1, kv in zip(subset, ret):
        assert k1 == kv[0]
        assert items[k1] == kv[1]


@given(dict_and_subset())  # pylint: disable=no-value-for-parameter
def test_iter_bounds_reverse(g):
    items, subset, lower_bound, upper_bound = g
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
