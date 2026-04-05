# Sequence Loader: Metadata Parser Task

This step extends `pipeline/sequence_loader.py` with scene metadata parsing.

Implemented in this task:

- parse `seqinfo.ini` for `seqLength`, `frameRate`, `imWidth`, and `imHeight`
- build a `Scene` payload through `get_scene_payload()`
- infer the dataset split from the sequence path
- load altitude from `data/visdrone_sequence_meta.json` when available
- fall back to stable defaults for fields that require later classifiers or frame analysis

Current fallback behavior:

- `altitude_m`: `50.0` with `altitude_source="estimated"` when lookup data is absent
- `weather`: `clear`
- `weather_source`: `default`
- `scene_type`: `urban`
- `time_of_day`: `daytime`

Still not implemented in this step:

- annotation parsing
- frame iteration and `FramePacket` generation
- visual inference for weather, scene type, or time-of-day
