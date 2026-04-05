# Sequence Loader Test Coverage

`tests/test_sequence_loader.py` now covers the completed `SequenceLoader` work:

- sequence discovery from supported VisDrone directory layouts
- optional `gt.txt` handling
- approved sequence manifest loading
- `seqinfo.ini` parsing and scene payload defaults
- frame iteration, frame skipping, and first-frame `scene_payload`
- explicit failure paths for invalid metadata and unreadable frames
- the current `NotImplementedError` behavior for deferred annotation parsing
