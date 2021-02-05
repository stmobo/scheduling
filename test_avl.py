from hypothesis import given, note, assume, strategies as st
from avl import AVLTree


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


@given(dict_and_key())  # pylint: disable=no-value-for-parameter
def test_insert_positive(givens):
    items, test_key = givens
    tree = AVLTree()

    for k, v in items.items():
        assert tree.insert(k, v) is None

    assert tree.get(test_key) == items[test_key]


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
