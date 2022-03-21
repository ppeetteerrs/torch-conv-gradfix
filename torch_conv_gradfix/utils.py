import re
from functools import total_ordering
from typing import Any, Collection, Literal, Tuple, TypeVar, Union, cast, overload

import torch

IntPair = Tuple[int, int]
IntPairOr = Union[int, Tuple[int, int]]

T = TypeVar("T")
CollectionOr = Union[T, Collection[T]]


@overload
def tuple_guard(item: CollectionOr[T]) -> Tuple[T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[2]
) -> Tuple[T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[3]
) -> Tuple[T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[4]
) -> Tuple[T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[5]
) -> Tuple[T, T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[6]
) -> Tuple[T, T, T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[7]
) -> Tuple[T, T, T, T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[8]
) -> Tuple[T, T, T, T, T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[9]
) -> Tuple[T, T, T, T, T, T, T, T, T]:  # pragma: no cover
    ...


@overload
def tuple_guard(
    item: CollectionOr[T], dim: Literal[10]
) -> Tuple[T, T, T, T, T, T, T, T, T, T]:  # pragma: no cover
    ...


def tuple_guard(item: CollectionOr[T], dim: int = 2) -> Tuple[T, ...]:
    """
    Checks input and returns a tuple of the specified length.
    """
    if isinstance(item, Collection):  # pragma: no cover
        item = cast(Collection[T], item)
        assert len(item) == dim, f"Input {item} should have length {dim}"
        return tuple(item)
    else:
        return (item,) * dim


@total_ordering
class Version:  # pragma: no cover
    def __init__(self, text: Union[int, str, float]) -> None:
        text = str(text)
        m = re.match(r"\D*?(\d+)\.?(\d*)\.?(\d*)", text)
        if m is None:
            raise Exception(f"{text} is not a valid version string")
        self.versions = [int(i) if i != "" else 0 for i in m.groups()]

    def __eq__(self, __o: Any) -> bool:
        if isinstance(__o, str):
            __o = Version(__o)
        if isinstance(__o, (float, int)):
            __o = Version(str(__o))

        if isinstance(__o, Version):
            return all(i == j for i, j in zip(self.versions, __o.versions))
        else:
            return False

    def __gt__(self, __o: Any) -> bool:
        if isinstance(__o, str):
            __o = Version(__o)
        if isinstance(__o, (float, int)):
            __o = Version(str(__o))

        if isinstance(__o, Version):
            for i, j in zip(self.versions, __o.versions):
                if i > j:
                    return True
                elif i < j:
                    return False
        return False

    def __repr__(self) -> str:
        return ".".join([str(i) for i in self.versions])


torch_version = Version(torch.__version__)
