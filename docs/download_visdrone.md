# VisDrone Download Script

The repository includes a download helper at:

`infra/download_visdrone.sh`

It places assets in the project-standard locations:

- `data/visdrone/` for extracted VisDrone archives
- `weights/` for the base YOLO checkpoint

## Default Behavior

With no arguments, the script downloads:

- `VisDrone2019-MOT-train`
- `VisDrone2019-MOT-val`
- `weights/yolov8m.pt`

## Optional Modes

```bash
bash infra/download_visdrone.sh --det
bash infra/download_visdrone.sh --weights
bash infra/download_visdrone.sh --all
```

`--all` adds the DET train, val, and test-dev archives under `data/visdrone/`.

## Notes

- The script prefers `venv/bin/python` when present.
- Dataset archives are downloaded to a temporary directory and extracted into `data/visdrone/`.
- `DRY_RUN=1` prints the target paths without downloading anything.
