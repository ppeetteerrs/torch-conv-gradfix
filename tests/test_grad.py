from functools import partial
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.autograd import gradcheck, gradgradcheck
from torch_conv_gradfix import (conv2d, conv_transpose2d, disable, enable,
                                no_weight_grad)

N = 2
SIZE = 8
IN_CHANNEL = 3
OUT_CHANNEL = 8
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 2
DILATION = 1
GROUPS = 1

conv2d_test = partial(
    conv2d,
    stride=STRIDE,
    padding=PADDING,
    dilation=DILATION,
    groups=GROUPS,
)

conv_transpose2d_test = partial(
    conv_transpose2d,
    stride=STRIDE,
    padding=PADDING,
    dilation=DILATION,
    groups=GROUPS,
)


def get_tensors(transpose: bool, device: str) -> Tuple[Tensor, Tensor, Tensor]:
    input = torch.randn(
        N, IN_CHANNEL, SIZE, SIZE, dtype=torch.double, requires_grad=True, device=device
    )
    if not transpose:
        weight = torch.randn(
            OUT_CHANNEL,
            IN_CHANNEL,
            KERNEL_SIZE,
            KERNEL_SIZE,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
    else:
        weight = torch.randn(
            IN_CHANNEL,
            OUT_CHANNEL,
            KERNEL_SIZE,
            KERNEL_SIZE,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
    bias = torch.randn(
        OUT_CHANNEL, dtype=torch.double, requires_grad=True, device=device
    )
    return input, weight, bias


def fn_test(
    fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    inputs: Tuple[Tensor, Tensor, Tensor],
):
    input, weight, bias = inputs

    # Check correctness
    assert gradcheck(fn, (input, weight, bias))
    assert gradgradcheck(fn, (input, weight, bias))

    # Check no_weight_grad
    input.grad = weight.grad = bias.grad = None
    enable()
    output: Tensor = fn(input, weight, bias)
    with no_weight_grad():
        output.sum().backward()
        if input.device.type == "cpu":
            test_weight = weight.grad is not None
        else:
            test_weight = weight.grad is None
        assert input.grad is not None and test_weight and bias.grad is not None

    # Check disable
    input.grad = weight.grad = bias.grad = None
    disable()
    output: Tensor = fn(input, weight, bias)
    with no_weight_grad():
        output.sum().backward()
        assert (
            input.grad is not None and weight.grad is not None and bias.grad is not None
        )


def test_cpu():
    fn_test(conv2d_test, get_tensors(False, "cpu"))
    fn_test(conv_transpose2d_test, get_tensors(True, "cpu"))


def test_cuda():
    fn_test(conv2d_test, get_tensors(False, "cuda"))
    fn_test(conv_transpose2d_test, get_tensors(True, "cuda"))


if __name__ == "__main__":
    test_cpu()
    test_cuda()
