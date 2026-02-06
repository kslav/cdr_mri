#!/usr/bin/env python
"""
Create a simulated MRI dataset for testing the native complex number implementation.

This script generates:
- Complex-valued images (Shepp-Logan phantom or random texture)
- Multi-coil sensitivity maps (smooth Gaussian coils)
- Undersampling masks (random or regular patterns)
- K-space data from forward encoding

Output: HDF5 file compatible with MultiChannelMRIDataset
"""

import numpy as np
import h5py
import argparse
from scipy.ndimage import gaussian_filter


def create_shepp_logan_phantom(size=64):
    """
    Create a simplified 2D Shepp-Logan phantom.

    Args:
        size: Image dimension (square image)

    Returns:
        Complex-valued 2D image
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    # Create phantom
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)

    # Add some phase to make it complex
    phase = np.random.randn(size, size) * 0.1
    phantom_complex = phantom * np.exp(1j * phase)

    return phantom_complex.astype(np.complex64)


def create_random_texture_phantom(size=64, num_blobs=20):
    """
    Create a random texture phantom with Gaussian blobs.

    Args:
        size: Image dimension (square image)
        num_blobs: Number of random blobs to add

    Returns:
        Complex-valued 2D image
    """
    img = np.zeros((size, size), dtype=np.complex64)

    # Add random Gaussian blobs
    for _ in range(num_blobs):
        y, x = np.random.randint(0, size, 2)
        sigma = np.random.uniform(2, 8)
        amplitude = np.random.uniform(0.1, 1.0)

        # Create blob
        y_grid, x_grid = np.ogrid[:size, :size]
        blob = amplitude * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))

        # Add with random phase
        phase = np.random.uniform(0, 2*np.pi)
        img += blob * np.exp(1j * phase)

    # Smooth slightly
    img_real = gaussian_filter(img.real, sigma=1.0)
    img_imag = gaussian_filter(img.imag, sigma=1.0)
    img = img_real + 1j * img_imag

    # Normalize
    img = img / np.abs(img).max()

    return img.astype(np.complex64)


def create_sensitivity_maps(img_shape, num_coils=4, smoothness=10.0):
    """
    Create smooth sensitivity maps for multi-coil MRI.

    Args:
        img_shape: Tuple of (height, width)
        num_coils: Number of coil elements
        smoothness: Gaussian smoothing sigma (higher = smoother)

    Returns:
        Sensitivity maps with shape (num_coils, height, width)
    """
    height, width = img_shape
    maps = np.zeros((num_coils, height, width), dtype=np.complex64)

    # Create grid
    y, x = np.ogrid[:height, :width]
    y = (y - height/2) / (height/2)
    x = (x - width/2) / (width/2)

    # Place coils in circular arrangement
    for i in range(num_coils):
        angle = 2 * np.pi * i / num_coils
        # Coil center position on a circle
        coil_y = 0.8 * np.cos(angle)
        coil_x = 0.8 * np.sin(angle)

        # Distance from coil center
        dist = np.sqrt((x - coil_x)**2 + (y - coil_y)**2)

        # Gaussian sensitivity profile
        sensitivity = np.exp(-dist**2 / (2 * (0.5)**2))

        # Add smooth phase variation
        phase = angle + 0.1 * (x * np.cos(angle) + y * np.sin(angle))

        maps[i] = sensitivity * np.exp(1j * phase)

    # Smooth the maps
    for i in range(num_coils):
        maps[i].real[:] = gaussian_filter(maps[i].real, sigma=smoothness)
        maps[i].imag[:] = gaussian_filter(maps[i].imag, sigma=smoothness)

    # Normalize: sum of squared magnitudes should be ~1
    sos = np.sqrt(np.sum(np.abs(maps)**2, axis=0))
    maps = maps / (sos[None, :, :] + 1e-8)

    return maps.astype(np.complex64)


def create_undersampling_mask(img_shape, acceleration=4, pattern='random'):
    """
    Create an undersampling mask for compressed sensing.

    Args:
        img_shape: Tuple of (height, width)
        acceleration: Acceleration factor (e.g., 4 means keep 25% of k-space)
        pattern: 'random', 'uniform', or 'center' for different sampling patterns

    Returns:
        Binary mask with shape (height, width)
    """
    height, width = img_shape
    mask = np.zeros(img_shape, dtype=np.float32)

    if pattern == 'random':
        # Random k-space lines (1D undersampling along phase encode direction)
        num_lines = int(height / acceleration)

        # Always sample center of k-space
        center_fraction = 0.08  # 8% of center
        num_center = int(height * center_fraction)
        center_start = height // 2 - num_center // 2
        mask[center_start:center_start + num_center, :] = 1.0

        # Randomly sample remaining lines
        num_random = num_lines - num_center
        remaining_lines = [i for i in range(height)
                          if i < center_start or i >= center_start + num_center]
        sampled_lines = np.random.choice(remaining_lines, num_random, replace=False)
        mask[sampled_lines, :] = 1.0

    elif pattern == 'uniform':
        # Uniform undersampling
        step = int(acceleration)
        mask[::step, :] = 1.0

        # Always include center
        center = height // 2
        mask[center-2:center+2, :] = 1.0

    elif pattern == 'center':
        # Only keep center of k-space (for testing)
        keep_fraction = 1.0 / acceleration
        keep_size = int(height * keep_fraction)
        start = height // 2 - keep_size // 2
        mask[start:start + keep_size, :] = 1.0

    return mask.astype(np.float32)


def fft2c_numpy(x):
    """Centered 2D FFT for numpy arrays."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def ifft2c_numpy(k):
    """Centered 2D IFFT for numpy arrays."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def create_dataset(output_file, num_images=10, img_size=64, num_coils=4,
                   acceleration=4, noise_level=0.0, phantom_type='shepp-logan',
                   mask_pattern='random'):
    """
    Create a complete simulated MRI dataset.

    Args:
        output_file: Output HDF5 filename
        num_images: Number of images in dataset
        img_size: Image dimension (square)
        num_coils: Number of coil elements
        acceleration: Undersampling acceleration factor
        noise_level: Standard deviation of complex Gaussian noise
        phantom_type: 'shepp-logan' or 'random'
        mask_pattern: 'random', 'uniform', or 'center'
    """
    print(f"Creating simulated MRI dataset:")
    print(f"  Output file: {output_file}")
    print(f"  Number of images: {num_images}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of coils: {num_coils}")
    print(f"  Acceleration: {acceleration}x")
    print(f"  Noise level: {noise_level}")
    print(f"  Phantom type: {phantom_type}")
    print(f"  Mask pattern: {mask_pattern}")
    print()

    # Create arrays to store data
    imgs = np.zeros((num_images, img_size, img_size), dtype=np.complex64)
    maps = np.zeros((num_images, num_coils, img_size, img_size), dtype=np.complex64)
    masks = np.zeros((num_images, img_size, img_size), dtype=np.float32)
    ksp = np.zeros((num_images, num_coils, img_size, img_size), dtype=np.complex64)

    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...", end='\r')

        # Create phantom
        if phantom_type == 'shepp-logan':
            img = create_shepp_logan_phantom(img_size)
        else:
            img = create_random_texture_phantom(img_size)

        # Create sensitivity maps
        coil_maps = create_sensitivity_maps((img_size, img_size), num_coils)

        # Create undersampling mask
        mask = create_undersampling_mask((img_size, img_size), acceleration, mask_pattern)

        # Forward encoding: image -> coil images -> k-space
        coil_imgs = img[None, :, :] * coil_maps  # (num_coils, H, W)
        kspace_full = fft2c_numpy(coil_imgs)  # (num_coils, H, W)

        # Add noise if specified
        if noise_level > 0:
            noise = (np.random.randn(*kspace_full.shape) +
                    1j * np.random.randn(*kspace_full.shape)) * noise_level / np.sqrt(2)
            kspace_full = kspace_full + noise

        # Apply undersampling mask
        # Note: mask is (H, W), need to broadcast to (num_coils, H, W)

        imgs[i] = img
        maps[i] = coil_maps
        masks[i] = mask
        ksp[i] = kspace_full

    print(f"\nGenerating image {num_images}/{num_images}... Done!")

    # Calculate statistics
    print("\nDataset statistics:")
    print(f"  Images: shape={imgs.shape}, dtype={imgs.dtype}")
    print(f"  Maps: shape={maps.shape}, dtype={maps.dtype}")
    print(f"  Masks: shape={masks.shape}, dtype={masks.dtype}, "
          f"sampling={masks.mean():.1%}")
    print(f"  K-space: shape={ksp.shape}, dtype={ksp.dtype}")

    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('imgs', data=imgs, dtype=np.complex64)
        f.create_dataset('maps', data=maps, dtype=np.complex64)
        f.create_dataset('masks', data=masks, dtype=np.float32)
        f.create_dataset('ksp', data=ksp, dtype=np.complex64)

        # Add metadata as attributes
        f.attrs['num_images'] = num_images
        f.attrs['img_size'] = img_size
        f.attrs['num_coils'] = num_coils
        f.attrs['acceleration'] = acceleration
        f.attrs['noise_level'] = noise_level
        f.attrs['phantom_type'] = phantom_type
        f.attrs['mask_pattern'] = mask_pattern

    print(f"✓ Dataset saved successfully!")
    print(f"  File size: {h5py.File(output_file, 'r')['imgs'].nbytes * 4 / 1024**2:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create simulated MRI dataset for testing')
    parser.add_argument('--output', type=str, default='data/simulated_mri_data.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to generate')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Image size (square)')
    parser.add_argument('--num-coils', type=int, default=4,
                       help='Number of coil elements')
    parser.add_argument('--acceleration', type=float, default=4.0,
                       help='Acceleration factor')
    parser.add_argument('--noise-level', type=float, default=0.01,
                       help='Noise standard deviation')
    parser.add_argument('--phantom-type', type=str, default='shepp-logan',
                       choices=['shepp-logan', 'random'],
                       help='Type of phantom to generate')
    parser.add_argument('--mask-pattern', type=str, default='random',
                       choices=['random', 'uniform', 'center'],
                       help='Undersampling pattern')

    args = parser.parse_args()

    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                exist_ok=True)

    create_dataset(
        output_file=args.output,
        num_images=args.num_images,
        img_size=args.img_size,
        num_coils=args.num_coils,
        acceleration=args.acceleration,
        noise_level=args.noise_level,
        phantom_type=args.phantom_type,
        mask_pattern=args.mask_pattern
    )
