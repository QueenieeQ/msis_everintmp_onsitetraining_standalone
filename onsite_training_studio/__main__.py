"""Run: ``python -m onsite_training_studio [--workspace DIR] [--port 8766]``."""

from __future__ import annotations

import argparse

from onsite_training_studio import paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Onsite Training Studio (annotate shots + train YOLO models)")
    parser.add_argument("--workspace", help=f"data folder (default: ${paths.ENV_VAR} or <project>/workspace)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    if args.workspace:
        paths.set_workspace(args.workspace)
    paths.registration_root().mkdir(parents=True, exist_ok=True)
    print(f"workspace: {paths.workspace_root()}")

    import uvicorn

    uvicorn.run(
        "onsite_training_studio.studio.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
