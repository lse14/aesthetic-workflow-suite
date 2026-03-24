# infer_ui

Independent batch inference tool for:

- 4-dimension regression scoring (`aesthetic`, `composition`, `color`, `sexual`)
- special-image detection via `in_domain` classification head
- auto-tagging unsuitable/special images
- organizing images into folders by `dimension + score bucket`

## Quick start (WebUI)

```bat
cd infer_ui
start.bat
```

WebUI opens in browser and supports one-click run/stop with real-time logs.

## Portable packaging (zip `infer_ui` only)

1. Compress the whole `infer_ui` folder.
2. Send model file (`.safetensors`) separately.
3. Receiver extracts zip, runs `start.bat`, then fills checkpoint model path in WebUI.

Optional (recommended): run `prefetch_jtp3.bat` first to pre-download JTP-3 base into `infer_ui/_models/repos/RedRocket__JTP-3`.

## CLI mode

```bat
cd infer_ui
py -3 run.py --config config.yaml --input-dir D:\images --output-dir D:\out --special-threshold 0.45 --organize on
```

## Minimal mode (only model path)

```bat
cd infer_ui
py -3 run.py --model D:\vscode\vibecode\outputs\fusion_1k_baseline\best.safetensors
```

Defaults used when omitted:

- `input_dir`: `data/infer_images` (then fallback to checkpoint sibling `images`)
- `output_dir`: `outputs/infer_run` (or checkpoint sibling `infer_run` when empty)

## Run with overrides

```bat
cd infer_ui
py -3 run.py --config config.yaml --input-dir D:\images --output-dir D:\out --special-threshold 0.45 --organize on
```

## Outputs

- `predictions.jsonl` / `predictions.csv`: per-image scores + tags
- `summary.json`: run summary
- `organized/`: optional folder organization output
  - `organized/special/<dimension>/score_<1..5>/...`
  - `organized/in_domain/<dimension>/score_<1..5>/...`

## Notes

- `special_tag=1` means `in_domain_prob < special_threshold`.
- If checkpoint has no classification head, `special_tag` defaults to `0` and `special_reason=no_cls_head`.
- `organize.mode=move` with multiple dimensions automatically falls back to `copy` for safety.
