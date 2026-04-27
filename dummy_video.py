import cv2
import numpy as np
from pathlib import Path

# Create data/dummy.mp4
out_path = Path("data/dummy.mp4")
out_path.parent.mkdir(exist_ok=True, parents=True)

writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    writer.write(frame)
writer.release()
print(f"Created {out_path.absolute()}")
