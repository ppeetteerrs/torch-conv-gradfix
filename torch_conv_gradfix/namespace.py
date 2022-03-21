import sys
from types import ModuleType
from typing import ContextManager, Optional, Type, cast

from torch import Tensor

from torch_conv_gradfix.conv_gradfix import ConvGradFix
from torch_conv_gradfix.utils import IntPairOr


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: IntPairOr = 1,
    padding: IntPairOr = 0,
    dilation: IntPairOr = 1,
    groups: int = 1,
) -> Tensor:  # pragma: no cover
    ...


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: IntPairOr = 1,
    padding: IntPairOr = 0,
    output_padding: IntPairOr = 0,
    dilation: IntPairOr = 1,
    groups: int = 1,
) -> Tensor:  # pragma: no cover
    ...


def enable():  # pragma: no cover
    ...


def disable():  # pragma: no cover
    ...


sys.modules[__name__] = cast(ModuleType, ConvGradFix)

no_weight_grad: Type[ContextManager[None]]
