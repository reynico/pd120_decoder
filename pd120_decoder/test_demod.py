import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from demod import (
    create_hilbert,
    create_analytica,
    boundary,
    hpf,
    process_audio,
    decode
)
from utils import mapper, write_px


@pytest.fixture
def test_image():
    return Image.new('YCbCr', (10, 10), "white")


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_create_hilbert():
    # Test with atten < 21
    hilbert_low = create_hilbert(20, np.pi/2)
    assert len(hilbert_low) > 0
    assert isinstance(hilbert_low, np.ndarray)

    # Test with 21 < atten < 50
    hilbert_mid = create_hilbert(35, np.pi/2)
    assert len(hilbert_mid) > 0
    assert isinstance(hilbert_mid, np.ndarray)

    # Test with atten > 50
    hilbert_high = create_hilbert(60, np.pi/2)
    assert len(hilbert_high) > 0
    assert isinstance(hilbert_high, np.ndarray)


def test_boundary():
    # Test lower bound
    assert boundary(1400) == 1500
    assert boundary(1500) == 1500

    # Test upper bound
    assert boundary(2300) == 2300

    # Test middle value
    assert boundary(2000) == 2000


def test_create_analytica():
    # Create sample data and filter
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filter = create_hilbert(40, np.pi/2)

    result = create_analytica(data, filter)
    assert len(result) == len(data) + len(filter) - 1
    assert isinstance(result[0], complex)


def test_hpf():
    # Create sample data
    fs = 44100
    t = np.linspace(0, 1, fs)
    # Create a signal with both low and high frequencies
    data = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 1000 * t)

    filtered = hpf(data, fs)
    assert len(filtered) == len(data)
    assert isinstance(filtered, np.ndarray)


def test_write_px(test_image):
    # Test luminance writing
    write_px(test_image, 0, 1, "lum", 1900)
    pixel = test_image.getpixel((0, 0))
    assert pixel[0] == mapper(1900)

    # Test Cr writing
    write_px(test_image, 0, 1, "cr", 1900)
    pixel = test_image.getpixel((0, 0))
    assert pixel[2] == mapper(1900)

    # Test Cb writing
    write_px(test_image, 0, 1, "cb", 1900)
    pixel = test_image.getpixel((0, 0))
    assert pixel[1] == mapper(1900)


def test_decode_with_sample_data(temp_output_dir):
    # Create sample data
    fs = 44100
    samples = np.ones(100000) * 1000  # Create dummy samples

    # Test decode function
    img = decode(0, samples, fs, temp_output_dir, "test")
    assert isinstance(img, Image.Image)
    assert img.size == (640, 496)
    assert img.mode == 'YCbCr'


def compare_images(img1, img2, tolerance=10):
    """
    Compare two images pixel by pixel, allowing for small differences.

    Args:
        img1, img2: PIL Image objects
        tolerance: Maximum allowed difference per channel per pixel

    Returns:
        bool: True if images are similar within tolerance
        str: Description of differences if images differ significantly
    """
    if img1.size != img2.size:
        return False, f"Size mismatch: {img1.size} != {img2.size}"

    if img1.mode != img2.mode:
        # Convert both to RGB for comparison if modes don't match
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

    # Convert images to numpy arrays for easier comparison
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Calculate differences
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        # Calculate percentage of pixels that differ significantly
        significant_diff_pixels = np.sum(diff > tolerance)
        total_pixels = diff.size
        diff_percentage = (significant_diff_pixels / total_pixels) * 100

        return False, (f"Images differ: max difference = {max_diff}, "
                       f"{diff_percentage:.2f}% pixels differ significantly")

    return True, "Images are similar within tolerance"


def test_process_audio(temp_output_dir):
    root_dir = Path(__file__).resolve().parent.parent
    input_wav = root_dir / "examples/iss-20201225-002100-short.wav"
    reference_image_path = root_dir / "examples/iss-20201225-002100-short.wav-7766.png"

    assert Path(input_wav).exists(), f"Input WAV file not found: {input_wav}"
    assert Path(reference_image_path).exists(), f"Reference image not found: {reference_image_path}"

    # Process the audio file
    results = process_audio(input_wav, temp_output_dir)
    assert isinstance(results, list), "Expected a list of images"
    assert len(results) > 0, "No images were generated"

    reference_img = Image.open(reference_image_path)

    is_similar, message = compare_images(results[0], reference_img)

    if not is_similar:
        debug_output = Path(temp_output_dir) / "debug_output.png"
        results[0].save(debug_output)
        print(f"Debug image saved to: {debug_output}")

    assert is_similar, f"Generated image differs from reference: {message}"
