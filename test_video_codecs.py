import cv2
import numpy as np
import tempfile
from pathlib import Path

frame = np.zeros((100, 100, 3), dtype=np.uint8)

codecs = [
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("h264", ".mp4"),
    ("vp80", ".webm"),
    ("vp90", ".webm"),
]

for fourcc_str, ext in codecs:
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        path = temp_file.name
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*fourcc_str),
        10.0,
        (100, 100),
    )
    opened = writer.isOpened()
    if opened:
        writer.write(frame)
    writer.release()
    
    size = Path(path).stat().st_size
    print(f"Codec: {fourcc_str}, isOpened: {opened}, File Size: {size}")
    Path(path).unlink()
