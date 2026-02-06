#!/usr/bin/env python
"""
Test script to verify PyTorch native complex number support in CDR-MRI.

This script tests:
1. Complex utility functions (zmul, zconj, zabs)
2. FFT forward and adjoint operations
3. Sensitivity map operations
4. Full SENSE forward/adjoint operators
5. Comparison between native complex and 2-channel real implementations
"""

import numpy as np
import torch
import sys
sys.path.insert(0, 'deepinpy')

from deepinpy.utils import complex as cp
from deepinpy.forwards.mcmri.mcmri import (
    fft_forw, fft_adj, maps_forw, maps_adj,
    mask_forw, sense_forw, sense_adj
)

print("="*80)
print("Testing PyTorch Native Complex Number Support")
print("="*80)
print(f"\nPyTorch version: {torch.__version__}")
print(f"Complex support available: {hasattr(torch, 'complex64')}")
print()

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
batch_size = 2
num_coils = 4
img_height = 32
img_width = 32

print(f"Test Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Number of coils: {num_coils}")
print(f"  Image size: {img_height} x {img_width}")
print()

# ============================================================================
# 1. Create simulated test data
# ============================================================================
print("-"*80)
print("1. Creating simulated MRI data")
print("-"*80)

# Create complex-valued image
img_real = np.random.randn(batch_size, img_height, img_width).astype(np.float32)
img_imag = np.random.randn(batch_size, img_height, img_width).astype(np.float32)
img_np = img_real + 1j * img_imag

# Create sensitivity maps for each coil
maps_real = np.random.randn(batch_size, num_coils, img_height, img_width).astype(np.float32) * 0.5
maps_imag = np.random.randn(batch_size, num_coils, img_height, img_width).astype(np.float32) * 0.5
maps_np = maps_real + 1j * maps_imag
# Normalize maps
maps_np = maps_np / (np.abs(maps_np).sum(axis=1, keepdims=True) + 1e-8)

# Create undersampling mask (e.g., 4x acceleration with random lines)
mask_np = np.zeros((batch_size, img_height, img_width), dtype=np.float32)
# Sample 25% of lines (4x acceleration)
num_lines_to_sample = img_height // 4
for b in range(batch_size):
    sampled_lines = np.random.choice(img_height, num_lines_to_sample, replace=False)
    mask_np[b, sampled_lines, :] = 1.0

print(f"✓ Created image with shape: {img_np.shape}")
print(f"✓ Created sensitivity maps with shape: {maps_np.shape}")
print(f"✓ Created undersampling mask with {mask_np.sum()}/{mask_np.size} samples")
print(f"  (acceleration factor: ~{mask_np.size / mask_np.sum():.1f}x)")
print()

# ============================================================================
# 2. Test complex utility functions
# ============================================================================
print("-"*80)
print("2. Testing complex utility functions")
print("-"*80)

# Convert to both formats
img_complex = torch.from_numpy(img_np.astype(np.complex64))
img_2ch = torch.from_numpy(cp.c2r(img_np))

maps_complex = torch.from_numpy(maps_np.astype(np.complex64))
maps_2ch = torch.from_numpy(cp.c2r(maps_np))

print(f"Native complex tensor shape: {img_complex.shape}, dtype: {img_complex.dtype}")
print(f"2-channel real tensor shape: {img_2ch.shape}, dtype: {img_2ch.dtype}")
print()

# Test zmul (complex multiplication)
print("Testing zmul (complex multiplication):")
a_complex = torch.randn(4, 4, dtype=torch.complex64)
b_complex = torch.randn(4, 4, dtype=torch.complex64)
a_2ch = cp.complex_to_real2ch(a_complex)
b_2ch = cp.complex_to_real2ch(b_complex)

result_native = cp.zmul(a_complex, b_complex)
result_2ch = cp.zmul(a_2ch, b_2ch)
result_2ch_as_complex = cp.real2ch_to_complex(result_2ch)

error = torch.abs(result_native - result_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-6, "zmul produces different results!"
print("  ✓ zmul works correctly for both formats")
print()

# Test zconj (complex conjugate)
print("Testing zconj (complex conjugate):")
result_native = cp.zconj(a_complex)
result_2ch = cp.zconj(a_2ch)
result_2ch_as_complex = cp.real2ch_to_complex(result_2ch)

error = torch.abs(result_native - result_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-6, "zconj produces different results!"
print("  ✓ zconj works correctly for both formats")
print()

# Test zabs (complex magnitude)
print("Testing zabs (complex magnitude):")
result_native = cp.zabs(a_complex)
result_2ch = cp.zabs(a_2ch)

error = torch.abs(result_native - result_2ch).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-6, "zabs produces different results!"
print("  ✓ zabs works correctly for both formats")
print()

# ============================================================================
# 3. Test FFT operations
# ============================================================================
print("-"*80)
print("3. Testing FFT operations")
print("-"*80)

# Test forward FFT
print("Testing fft_forw (forward FFT):")
fft_result_native = fft_forw(img_complex, ndim=2)
fft_result_2ch = fft_forw(img_2ch, ndim=2)
fft_result_2ch_as_complex = cp.real2ch_to_complex(fft_result_2ch)

error = torch.abs(fft_result_native - fft_result_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-5, "fft_forw produces different results!"
print("  ✓ fft_forw works correctly for both formats")
print()

# Test adjoint (inverse) FFT
print("Testing fft_adj (adjoint FFT):")
ifft_result_native = fft_adj(fft_result_native, ndim=2)
ifft_result_2ch = fft_adj(fft_result_2ch, ndim=2)
ifft_result_2ch_as_complex = cp.real2ch_to_complex(ifft_result_2ch)

error = torch.abs(ifft_result_native - ifft_result_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-5, "fft_adj produces different results!"
print("  ✓ fft_adj works correctly for both formats")

# Check FFT reconstruction (should get back original image)
recon_error_native = torch.abs(img_complex - ifft_result_native).max().item()
recon_error_2ch = torch.abs(img_2ch - ifft_result_2ch).max().item()
print(f"  FFT round-trip error (native): {recon_error_native:.2e}")
print(f"  FFT round-trip error (2-ch): {recon_error_2ch:.2e}")
assert recon_error_native < 1e-5, "FFT round-trip failed for native complex!"
assert recon_error_2ch < 1e-5, "FFT round-trip failed for 2-channel!"
print("  ✓ FFT round-trip successful")
print()

# ============================================================================
# 4. Test sensitivity map operations
# ============================================================================
print("-"*80)
print("4. Testing sensitivity map operations")
print("-"*80)

# Test maps_forw (sensitivity encoding)
print("Testing maps_forw (sensitivity encoding):")
coil_imgs_native = maps_forw(img_complex, maps_complex)
coil_imgs_2ch = maps_forw(img_2ch, maps_2ch)
coil_imgs_2ch_as_complex = cp.real2ch_to_complex(coil_imgs_2ch)

error = torch.abs(coil_imgs_native - coil_imgs_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
print(f"  Output shape (native): {coil_imgs_native.shape}")
print(f"  Output shape (2-ch): {coil_imgs_2ch.shape}")
assert error < 1e-5, "maps_forw produces different results!"
print("  ✓ maps_forw works correctly for both formats")
print()

# Test maps_adj (coil combination)
print("Testing maps_adj (coil combination):")
combined_img_native = maps_adj(coil_imgs_native, maps_complex)
combined_img_2ch = maps_adj(coil_imgs_2ch, maps_2ch)
combined_img_2ch_as_complex = cp.real2ch_to_complex(combined_img_2ch)

error = torch.abs(combined_img_native - combined_img_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
print(f"  Output shape (native): {combined_img_native.shape}")
print(f"  Output shape (2-ch): {combined_img_2ch.shape}")
assert error < 1e-5, "maps_adj produces different results!"
print("  ✓ maps_adj works correctly for both formats")
print()

# ============================================================================
# 5. Test mask operations
# ============================================================================
print("-"*80)
print("5. Testing mask operations")
print("-"*80)

mask_torch = torch.from_numpy(mask_np)

# Create some dummy k-space data
kspace_native = torch.randn(batch_size, num_coils, img_height, img_width, dtype=torch.complex64)
kspace_2ch = cp.complex_to_real2ch(kspace_native)

print("Testing mask_forw (k-space masking):")
masked_native = mask_forw(kspace_native, mask_torch)
masked_2ch = mask_forw(kspace_2ch, mask_torch)
masked_2ch_as_complex = cp.real2ch_to_complex(masked_2ch)

error = torch.abs(masked_native - masked_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
assert error < 1e-6, "mask_forw produces different results!"
print("  ✓ mask_forw works correctly for both formats")
print()

# ============================================================================
# 6. Test full SENSE forward and adjoint
# ============================================================================
print("-"*80)
print("6. Testing full SENSE forward and adjoint operators")
print("-"*80)

# Test SENSE forward (image -> undersampled k-space)
print("Testing sense_forw (full forward operator):")
kspace_native = sense_forw(img_complex, maps_complex, mask_torch, ndim=2)
kspace_2ch = sense_forw(img_2ch, maps_2ch, mask_torch, ndim=2)
kspace_2ch_as_complex = cp.real2ch_to_complex(kspace_2ch)

error = torch.abs(kspace_native - kspace_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
print(f"  K-space shape (native): {kspace_native.shape}")
print(f"  K-space shape (2-ch): {kspace_2ch.shape}")
assert error < 1e-4, "sense_forw produces different results!"
print("  ✓ sense_forw works correctly for both formats")
print()

# Test SENSE adjoint (undersampled k-space -> image)
print("Testing sense_adj (adjoint operator):")
img_recon_native = sense_adj(kspace_native, maps_complex, mask_torch, ndim=2)
img_recon_2ch = sense_adj(kspace_2ch, maps_2ch, mask_torch, ndim=2)
img_recon_2ch_as_complex = cp.real2ch_to_complex(img_recon_2ch)

error = torch.abs(img_recon_native - img_recon_2ch_as_complex).max().item()
print(f"  Max error between native and 2-channel: {error:.2e}")
print(f"  Reconstructed image shape (native): {img_recon_native.shape}")
print(f"  Reconstructed image shape (2-ch): {img_recon_2ch.shape}")
assert error < 1e-4, "sense_adj produces different results!"
print("  ✓ sense_adj works correctly for both formats")
print()

# ============================================================================
# 7. Test adjoint property: <Ax, y> = <x, A*y>
# ============================================================================
print("-"*80)
print("7. Verifying adjoint property")
print("-"*80)

print("Checking adjoint property: <Ax, y> = <x, A*y>")

# Create random test data
x_native = torch.randn(batch_size, img_height, img_width, dtype=torch.complex64)
y_native = torch.randn(batch_size, num_coils, img_height, img_width, dtype=torch.complex64)

# Compute forward and adjoint
Ax = sense_forw(x_native, maps_complex, mask_torch, ndim=2)
Aty = sense_adj(y_native, maps_complex, mask_torch, ndim=2)

# Compute inner products
# <Ax, y> = sum(conj(Ax) * y)
inner_Ax_y = torch.sum(torch.conj(Ax) * y_native).real.item()
# <x, A*y> = sum(conj(x) * A*y)
inner_x_Aty = torch.sum(torch.conj(x_native) * Aty).real.item()

print(f"  <Ax, y>   = {inner_Ax_y:.6f}")
print(f"  <x, A*y>  = {inner_x_Aty:.6f}")
relative_error = abs(inner_Ax_y - inner_x_Aty) / (abs(inner_Ax_y) + 1e-8)
print(f"  Relative error: {relative_error:.2e}")
assert relative_error < 1e-5, "Adjoint property violated!"
print("  ✓ Adjoint property verified")
print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("All tests passed! ✓")
print("="*80)
print("\nSummary:")
print("  • Complex utility functions (zmul, zconj, zabs) work correctly")
print("  • FFT forward/adjoint operations work correctly")
print("  • Sensitivity map operations work correctly")
print("  • Masking operations work correctly")
print("  • Full SENSE forward/adjoint operators work correctly")
print("  • Adjoint property is satisfied")
print("  • Native complex and 2-channel formats produce identical results")
print("\nThe codebase has been successfully updated to support PyTorch's")
print("native complex number handling!")
print("="*80)
