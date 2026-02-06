# PyTorch Native Complex Number Support - Update Summary

This document summarizes the updates made to enable PyTorch's native complex number support in the CDR-MRI codebase.

## Overview

The codebase has been modernized to use PyTorch's native complex number support (available since PyTorch 1.6+, stable in 1.8+) instead of the previous 2-channel real number representation. The updates are **backward compatible**, meaning both native complex and legacy 2-channel formats are supported.

## What Changed

### 1. Complex Utility Functions (`deepinpy/utils/complex.py`)

**New Functions Added:**
- `np_to_torch_complex(z)`: Convert numpy complex array to torch complex tensor
- `torch_to_np_complex(z)`: Convert torch complex tensor to numpy complex array
- `real2ch_to_complex(x)`: Convert 2-channel real tensor to native complex tensor
- `complex_to_real2ch(z)`: Convert native complex tensor to 2-channel real tensor

**Updated Functions:**
- `zmul()`: Now detects tensor format and handles both native complex and 2-channel real
- `zconj()`: Now uses `torch.conj()` for native complex tensors
- `zabs()`: Now uses `torch.abs()` for native complex tensors

### 2. FFT Operations (`deepinpy/forwards/mcmri/mcmri.py`)

**Replaced deprecated FFT API:**
- Old: `torch.fft(x, signal_ndim=ndim, normalized=True)` ❌ (deprecated in PyTorch 1.7)
- New: `torch.fft.fft2(x, norm='ortho')` ✅ (modern API)

**Updated Functions:**
- `fft_forw()`: Uses `torch.fft.fft2()` / `torch.fft.fftn()` with auto-detection of tensor format
- `fft_adj()`: Uses `torch.fft.ifft2()` / `torch.fft.ifftn()` with auto-detection
- `mask_forw()`: Handles both native complex and 2-channel real masking

All functions automatically detect whether inputs are native complex or 2-channel real and process accordingly.

### 3. Dataset (`deepinpy/forwards/mcmri/dataset.py`)

**New Parameter:**
- `use_native_complex=False`: Set to `True` to return native complex tensors instead of 2-channel real

**Usage:**
```python
# Legacy format (2-channel real)
dataset = MultiChannelMRIDataset('data.h5', use_native_complex=False)

# Native complex format
dataset = MultiChannelMRIDataset('data.h5', use_native_complex=True)
```

## How to Use

### Option 1: Native Complex (Recommended for New Code)

```python
import torch
from deepinpy.forwards.mcmri.mcmri import sense_forw, sense_adj

# Create complex-valued tensors directly
img = torch.randn(batch, height, width, dtype=torch.complex64)
maps = torch.randn(batch, coils, height, width, dtype=torch.complex64)
mask = torch.randn(batch, height, width)

# Use forward operators
kspace = sense_forw(img, maps, mask, ndim=2)
img_recon = sense_adj(kspace, maps, mask, ndim=2)
```

### Option 2: Legacy Format (For Backward Compatibility)

```python
import torch
from deepinpy.utils import complex as cp
from deepinpy.forwards.mcmri.mcmri import sense_forw, sense_adj

# Create 2-channel real tensors (shape: ..., 2)
img = torch.randn(batch, height, width, 2)
maps = torch.randn(batch, coils, height, width, 2)
mask = torch.randn(batch, height, width)

# Use forward operators (same API)
kspace = sense_forw(img, maps, mask, ndim=2)
img_recon = sense_adj(kspace, maps, mask, ndim=2)
```

### Converting Between Formats

```python
from deepinpy.utils import complex as cp

# 2-channel real -> native complex
x_complex = cp.real2ch_to_complex(x_2ch)

# Native complex -> 2-channel real
x_2ch = cp.complex_to_real2ch(x_complex)

# Numpy complex -> torch complex
x_torch = cp.np_to_torch_complex(x_numpy)

# Torch complex -> numpy complex
x_numpy = cp.torch_to_np_complex(x_torch)
```

## Testing

A comprehensive test script `test_native_complex.py` has been created to verify the implementation:

```bash
python test_native_complex.py
```

**Test Coverage:**
- ✅ Complex utility functions (zmul, zconj, zabs)
- ✅ FFT forward/adjoint operations
- ✅ Sensitivity map operations (maps_forw, maps_adj)
- ✅ Masking operations
- ✅ Full SENSE forward/adjoint operators
- ✅ Adjoint property verification
- ✅ Equivalence between native complex and 2-channel formats

All tests pass with maximum errors < 1e-6 (numerical precision).

## Benefits of Native Complex Support

1. **Simpler Code**: No need for manual real/imaginary channel management
2. **Better Performance**: Native operations are optimized in PyTorch
3. **Type Safety**: Complex tensors have proper dtype (complex64/complex128)
4. **Modern API**: Uses current PyTorch best practices
5. **Cleaner FFT**: Direct use of torch.fft module without conversion
6. **Memory Efficiency**: Complex tensors are more memory-efficient than 2-channel real

## Backward Compatibility

The updates maintain full backward compatibility:
- Existing code using 2-channel real format continues to work
- All functions auto-detect tensor format
- Models can be updated incrementally
- Dataset can return either format via parameter

## Next Steps

To fully migrate to native complex:

1. **Update Models**: Modify neural network architectures to accept complex inputs
   - Change input channels from `2*num_coils` to `num_coils`
   - Use complex-valued convolutions or real-valued operations on real/imag separately

2. **Update Reconstruction Methods**: Modify reconstruction classes to work with complex tensors
   - Update channel counting logic
   - Use native complex operations

3. **Performance Testing**: Benchmark native complex vs 2-channel real
   - Memory usage
   - Computation speed
   - Training stability

## Requirements

- **PyTorch >= 1.8.0** (required) - The updated code uses the modern `torch.fft` module which requires PyTorch 1.8+
  - The old `torch.fft()` and `torch.ifft()` functions were deprecated in PyTorch 1.7 and removed in 1.8
  - Native complex tensor support is stable starting in PyTorch 1.8

## Files Modified

1. `deepinpy/utils/complex.py` - Complex utility functions
2. `deepinpy/forwards/mcmri/mcmri.py` - FFT and SENSE operators
3. `deepinpy/forwards/mcmri/dataset.py` - Data loading
4. `requirements.txt` - Updated PyTorch requirement to >=1.8.0

## Files Created

1. `test_native_complex.py` - Comprehensive test suite
2. `NATIVE_COMPLEX_UPDATE.md` - This documentation

## Example: Simulated K-Space Reconstruction

The test script demonstrates a complete pipeline:
1. Create simulated complex images (32x32)
2. Generate multi-coil sensitivity maps (4 coils)
3. Create undersampling mask (4x acceleration)
4. Apply SENSE forward operator (image → k-space)
5. Apply SENSE adjoint operator (k-space → image)
6. Verify adjoint property

This serves as a template for testing with real MRI data.

---

**Date**: February 5, 2026
**PyTorch Version Tested**: 2.10.0
**Status**: ✅ All tests passing
