from demeter.utils import Lazy


def test_lazy():
    iterable = iter([1, 2, 3])
    lazy = Lazy(iterable)
    assert list(lazy) == [1, 2, 3]
    assert list(lazy) == [1, 2, 3]
