# Sequence Loader: Dataset Loader Task

This task adds the dataset-loading foundation in `pipeline/sequence_loader.py`.

Implemented in this step:

- resolve a VisDrone MOT sequence from disk
- validate the basic directory structure
- collect ordered frame image paths
- expose `gt.txt` when present
- read approved sequence IDs from `sequences.json`-style manifests

Supported `data_root` forms:

- a direct `.../sequences` directory
- a split root such as `.../VisDrone2019-MOT-val`
- the broader `data/visdrone` root

Not implemented in this step:

- `seqinfo.ini` parsing into scene metadata
- ground-truth annotation parsing
- frame iteration / `FramePacket` emission

Those remain for the next unchecked tasks in `TASKS.md`.
