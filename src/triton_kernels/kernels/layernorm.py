#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
import os


@triton.jit
def rms_norm(
    x_ptr,
    output_ptr,
    weight_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    RMS normalization kernel.
    Assumes: n_elements <= BLOCK_SIZE (single iteration only)
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_elements
    
    # Generate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Pass 1: Compute sum of x^2
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    # Convert to float32 for computation
    x_f32 = x.to(tl.float32)
    sum_x_squared = tl.sum(x_f32 * x_f32)
    
    # Compute normalization factor
    mean_x_squared = sum_x_squared / n_elements
    rstd = 1.0 / tl.sqrt(mean_x_squared + EPS)
    
    # Pass 2: Normalize and apply weight
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    weight_f32 = weight.to(tl.float32)
    
    output = x_f32 * rstd * weight_f32
    # Convert back to original dtype (bfloat16)
    output_bf16 = output.to(x.dtype)
    
    tl.store(output_ptr + row_start + offsets, output_bf16, mask=mask)


# Variants with different block sizes and configurations
VARIANTS = [
    {'BLOCK_SIZE': 1024, 'EPS': 1e-5},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-5},
    {'BLOCK_SIZE': 4096, 'EPS': 1e-5},
    {'BLOCK_SIZE': 8192, 'EPS': 1e-5},
    {'BLOCK_SIZE': 1024, 'EPS': 1e-6},
    {'BLOCK_SIZE': 2048, 'EPS': 1e-6},
    {'BLOCK_SIZE': 4096, 'EPS': 1e-6},
    {'BLOCK_SIZE': 8192, 'EPS': 1e-6},
]