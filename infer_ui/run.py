import argparse
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch inference and auto-tagging with 4 regression heads + 1 cls head."
    )
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Alias of --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--special-threshold", type=float, default=None)
    parser.add_argument(
        "--organize",
        choices=["auto", "on", "off"],
        default="auto",
        help="Folder organization switch. auto=use config.",
    )
    args = parser.parse_args()

    from scripts.batch_infer import run_from_config

    overrides: dict[str, object] = {}
    ckpt = args.model if args.model is not None else args.checkpoint
    if ckpt is not None:
        overrides["checkpoint"] = ckpt
    if args.input_dir is not None:
        overrides["input_dir"] = args.input_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.special_threshold is not None:
        overrides["special_threshold"] = float(args.special_threshold)
    if args.organize == "on":
        overrides["organize.enabled"] = True
    elif args.organize == "off":
        overrides["organize.enabled"] = False

    run_from_config(args.config, overrides=overrides)


if __name__ == "__main__":
    main()
