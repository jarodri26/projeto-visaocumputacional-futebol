import pytest
from src.data.extract_frames import extract_frames
import tempfile, os

def test_extract_frames_no_file():
    with pytest.raises(FileNotFoundError):
        extract_frames('nonexistent.webm', '/tmp/out', sample_rate=1)
