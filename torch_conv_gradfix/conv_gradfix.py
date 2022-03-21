import contextlib
import warnings
from functools import lru_cache
from typing import (Any, Callable, Optional, Protocol, Sequence, Tuple, Type,
                    cast)

import torch
from torch import Tensor, autograd
from torch.backends import cudnn
from torch.nn import functional as F

from torch_conv_gradfix.utils import (IntPair, IntPairOr, torch_version,
                                      tuple_guard)


class EnableFlag:
    """
    Checks torch_conv_gradfix availability on init. Allows setting of global flag but also performs runtime check.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.static_okay = True

        # Check Pytorch Version (Not tf32 flag before PyTorch 1.7)
        if torch_version < 1.7:  # pragma: no cover
            warnings.warn(
                f"torch_conv_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d()."
            )
            self.static_okay = False

        # Check CUDNN availability
        if not cudnn.enabled:  # pragma: no cover
            self.static_okay = False

    def __call__(self, x: Tensor) -> bool:
        return x.device.type == "cuda" and self.enabled and self.static_okay


class Conv2dFn(Protocol):
    @staticmethod
    def apply(
        input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:  # pragma: no cover
        ...


def calc_output_padding(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    transpose: bool,
    weight_shape: Sequence[int],
    stride: IntPair,
    padding: IntPair,
    dilation: IntPair,
) -> Tuple[int, int]:  # pragma: no cover
    if transpose:
        return (0, 0)
    else:
        result = tuple(
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(2)
        )
        return cast(Tuple[int, int], result)


CudnnConvBackwardSignature = Callable[
    [Sequence[int], Tensor, Tensor, IntPair, IntPair, IntPair, int, bool, bool, bool],
    Tensor,
]  # pragma: no cover


class ConvGradFix:
    enabled = EnableFlag()
    _no_weight_grad = False

    @staticmethod
    @lru_cache(maxsize=None)
    def conv2d_factory(
        transpose: bool,
        weight_shape: Sequence[int],
        stride: IntPairOr,
        padding: IntPairOr,
        output_padding: IntPairOr,
        dilation: IntPairOr,
        groups: int,
    ) -> Type[Conv2dFn]:
        weight_shape = tuple(weight_shape)
        stride_pair = tuple_guard(stride)
        padding_pair = tuple_guard(padding)
        output_padding_pair = tuple_guard(output_padding)
        dilation_pair = tuple_guard(dilation)

        class Conv2d(autograd.Function):
            @staticmethod
            def forward(
                ctx: Any, input: Tensor, weight: Tensor, bias: Optional[Tensor]
            ) -> Tensor:
                if not transpose:
                    out = F.conv2d(
                        input=input,
                        weight=weight,
                        bias=bias,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                        groups=groups,
                    )

                else:
                    out = F.conv_transpose2d(
                        input=input,
                        weight=weight,
                        bias=bias,
                        output_padding=output_padding_pair,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                        groups=groups,
                    )

                ctx.save_for_backward(input, weight)

                return out

            @staticmethod
            def backward(
                ctx: Any, grad_output: Tensor
            ) -> Tuple[
                Optional[Tensor], Optional[Tensor], Optional[Tensor]
            ]:  # pragma: no cover
                # Extract context
                input, weight = ctx.saved_tensors
                grad_input, grad_weight, grad_bias = None, None, None

                # Calculate gradients if needed
                if ctx.needs_input_grad[0]:
                    # Gradient w.r.t. input
                    p = calc_output_padding(
                        input_shape=input.shape,
                        output_shape=grad_output.shape,
                        transpose=transpose,
                        weight_shape=weight_shape,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                    )
                    grad_input = ConvGradFix.conv2d_factory(
                        transpose=(not transpose),
                        weight_shape=weight_shape,
                        output_padding=p,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                        groups=groups,
                    ).apply(grad_output, weight, None)

                if ctx.needs_input_grad[1] and not ConvGradFix._no_weight_grad:
                    # Gradient w.r.t. conv weights
                    grad_weight = Conv2dGradWeight.apply(grad_output, input)

                if ctx.needs_input_grad[2]:
                    # Gradient w.r.t. bias
                    grad_bias = grad_output.sum((0, 2, 3))

                return grad_input, grad_weight, grad_bias

            apply: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]

        class Conv2dGradWeight(autograd.Function):  # pragma: no cover
            @staticmethod
            def forward(ctx: Any, grad_output: Tensor, input: Tensor) -> Tensor:
                """
                See https://fossies.org/linux/pytorch/aten/src/ATen/native/cudnn/ConvShared.cpp: 404
                for cudnn_convolution_backward_weight implementation
                """
                op = cast(
                    CudnnConvBackwardSignature,
                    torch._C._jit_get_operation(
                        "aten::cudnn_convolution_backward_weight"
                        if not transpose
                        else "aten::cudnn_convolution_transpose_backward_weight"
                    ),
                )
                """
                cudnn.allow_tf32 type annotation is recently added.
                It should have been added since PyTorch 1.7 though...
                See torch.backends.cudnn.__init__.py for the hacks that make this possible
                (class CudnnModule)
                """
                grad_weight = op(
                    weight_shape,
                    grad_output,
                    input,
                    padding_pair,
                    stride_pair,
                    dilation_pair,
                    groups,
                    cudnn.benchmark,
                    cudnn.deterministic,
                    torch._C._get_cudnn_allow_tf32(),
                )
                ctx.save_for_backward(grad_output, input)

                return grad_weight

            @staticmethod
            def backward(
                ctx: Any, grad_grad_weight: Tensor
            ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
                # Extract context
                grad_output, input = ctx.saved_tensors
                grad_grad_output, grad_grad_input = None, None

                # Calculate gradients if needed
                # Gradient w.r.t. gradient w.r.t. output
                if ctx.needs_input_grad[0]:
                    grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)

                # Gradient w.r.t. input
                if ctx.needs_input_grad[1]:
                    p = calc_output_padding(
                        input_shape=input.shape,
                        output_shape=grad_output.shape,
                        transpose=transpose,
                        weight_shape=weight_shape,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                    )
                    grad_grad_input = ConvGradFix.conv2d_factory(
                        transpose=(not transpose),
                        weight_shape=weight_shape,
                        output_padding=p,
                        stride=stride_pair,
                        padding=padding_pair,
                        dilation=dilation_pair,
                        groups=groups,
                    ).apply(grad_output, grad_grad_weight, None)

                return grad_grad_output, grad_grad_input

            apply: Callable[[Tensor, Tensor], Tensor]

        return Conv2d

    @staticmethod
    def conv2d(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: IntPairOr = 1,
        padding: IntPairOr = 0,
        dilation: IntPairOr = 1,
        groups: int = 1,
    ) -> Tensor:
        if ConvGradFix.enabled(input):
            return ConvGradFix.conv2d_factory(
                transpose=False,
                weight_shape=weight.shape,
                stride=stride,
                padding=padding,
                output_padding=0,
                dilation=dilation,
                groups=groups,
            ).apply(input, weight, bias)
        return F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @staticmethod
    def conv_transpose2d(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: IntPairOr = 1,
        padding: IntPairOr = 0,
        output_padding: IntPairOr = 0,
        dilation: IntPairOr = 1,
        groups: int = 1,
    ) -> Tensor:
        if ConvGradFix.enabled(input):
            return ConvGradFix.conv2d_factory(
                transpose=True,
                weight_shape=weight.shape,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            ).apply(input, weight, bias)

        return F.conv_transpose2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )

    @staticmethod
    @contextlib.contextmanager
    def no_weight_grad():
        old = ConvGradFix._no_weight_grad
        ConvGradFix._no_weight_grad = True
        yield
        ConvGradFix._no_weight_grad = old

    @staticmethod
    def enable() -> None:
        ConvGradFix.enabled.enabled = True

    @staticmethod
    def disable() -> None:
        ConvGradFix.enabled.enabled = False
