# Onsite Training Studio

Standalone tool to **annotate registration shots** (left / right / closer) and
**train the per-shot YOLO models**. It has no dependency on `vision_system_fw`,
model_1, MQTT or the GUI — copy this folder to any PC and run it.

## Install & run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # add a CUDA build of torch for GPU training
./run_studio.sh                          # http://0.0.0.0:8766
./run_studio.sh --workspace /data/onsite --port 8766
```

Requirements: Python 3.10+. `yolo26*` weights need **ultralytics >= 8.4**
(tested on 8.4.174; 8.3.x fails with `SPPF.__init__() ... 6 were given`).
YOLO weights (`yolo11s.pt`, ...) download on first use, or place the `.pt`
files in the folder you start the server from.

CPU-only PC: open `<workspace>/onsite_training.yaml` (created on the first
training run) and set `device` to `cpu` in the `roi`/`screw`/`patch` sections
(default is GPU `"0"`).

## Workspace (all data lives here)

Resolved from `--workspace`, else `$ONSITE_WORKSPACE`, else `<this folder>/workspace`.

```
<workspace>/
  Product_Registration_Information/<mpfront|mpback>/<realsense|omron|Top_Camera>/<product>/
      configs/<product>_<front|back>_registration_info.yaml   boxes (H, S, P, ...)
      images/<shot>.png
  output/<stage>/<camera>/<product>/<shot>/     training results (see below)
      weights/weight.pt                          screw-hole (H) model
      weights/patches_weights/hole_<i>/weight_hole_<i>.pt   per-hole patch models (H)
      weights/screw_head/screw_head_detection.pt             screw-head (S) model
      images/ labels/ patches/ config/ref_user_extracted.json
  output/<stage>/<camera>/<product>/shots_config.yaml        per-shot summary
  onsite_training.yaml   training parameters (epochs, batch, device, augmentation, ...)
  logs/  runs/  tmp/     training logs, ultralytics runs, scratch datasets
```

To bring an existing product over, copy its
`Product_Registration_Information/<stage>/<camera>/<product>` folder into the
workspace.

## Using the studio

1. **Create product** (stage, camera, name) and **Upload** an image per shot — or copy a product in.
2. Pick shot + label, draw boxes (drag), move/resize, **Save annotations**:
   - **H** = screw hole, **S** = screw head, P = ROI polygon box (R/X/Y reserved).
3. Choose what to train:
   - **Train shot**: all shots or one.
   - **Train model**: all models, or exactly one of *Screw hole (H)*, *Patches (H)*, *Screw head (S)*.
     A single-model run updates the shot in place and keeps its other weights; it
     fails if the shot has no boxes for that model.
   - **YOLO version**: config default, yolo26 / yolo11 / yolov8 (n, s, m, l, x).
4. **Trigger onsite training** (runs in the background; the result is polled).
   Optional: send output to a different folder; untick *run training* to only rebuild the reference tree.

## Using the results with model_1 (vision framework)

The output folder uses model_1's layout and the paths inside
`ref_user_extracted.json` are already `data/models/model_1/<stage>/<camera>/<product>/...`.
On the vision PC copy `output/<stage>/<camera>/<product>/` to
`vision_system_fw/data/models/model_1/<stage>/<camera>/<product>/` and merge
`shots_config.yaml`'s `shots:` block into that product's model_1 vision-params
file (this tool no longer edits those). model_1 reads the H model
(`weights/weight.pt`); the screw-head model is not consumed by model_1 yet.

## Layout of this project

```
onsite_training_studio/
  paths.py errors.py colors.py      workspace paths, 115-xxx error codes
  training/                         dataset generation + YOLO training + reference-tree builder
  studio/                           FastAPI app + static UI (annotate / train)
tests/                              pytest (python -m pytest tests)
```
Programmatic use: `from onsite_training_studio.training.model import build_onsite_training_data`.
