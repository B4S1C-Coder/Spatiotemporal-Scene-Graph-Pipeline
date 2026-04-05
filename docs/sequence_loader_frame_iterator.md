# Sequence Loader: Frame Iterator Task

This task adds `iter_frames()` to `pipeline/sequence_loader.py`.

Implemented in this step:

- ordered frame loading from `img1/*.jpg`
- `frame_skip` support
- basic letterboxing to `img_size`
- emission of `FramePacket`-shaped dictionaries
- `scene_payload` included only on the first emitted frame

Current behavior:

- `annotations` is `[]` when `gt.txt` exists and `None` when it does not
- `is_static` is currently hard-coded to `False`
- unreadable frames raise a `ValueError`

Still deferred to later work:

- parsed annotation sidecars
- quality-control flags
- richer static-scene detection
