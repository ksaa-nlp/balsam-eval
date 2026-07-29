"""Command-line interface for the Balsam evaluation runner."""

import argparse
import os
from typing import Optional

import run as runner


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI options and delegate evaluation to the standard runner."""
    parser = argparse.ArgumentParser(
        prog="balsam-eval",
        description="Evaluate one or more Balsam pool dataset files.",
    )
    parser.add_argument(
        "pool_files",
        nargs="*",
        metavar="FILE",
        help="Pool dataset JSON files (defaults to .tasks/*.json)",
    )
    parser.add_argument("--model", help="Model identifier (or set MODEL)")
    parser.add_argument("--adapter", help="lm-eval adapter id (or set ADAPTER)")
    parser.add_argument("--api-key", help="Model API key (or set API_KEY)")
    parser.add_argument("--base-url", help="Model API base URL (or set BASE_URL)")
    parser.add_argument(
        "--judge-model",
        help="Comma-separated judge model ids (or set JUDGE_MODEL)",
    )
    parser.add_argument(
        "--judge-provider",
        help="Comma-separated judge providers (or set JUDGE_PROVIDER)",
    )
    parser.add_argument(
        "--judge-api-key",
        help="Comma-separated judge API keys (or set JUDGE_API_KEY)",
    )
    args = parser.parse_args(argv)

    cli_env = {
        "MODEL": args.model,
        "ADAPTER": args.adapter,
        "API_KEY": args.api_key,
        "BASE_URL": args.base_url,
        "JUDGE_MODEL": args.judge_model,
        "JUDGE_PROVIDER": args.judge_provider,
        "JUDGE_API_KEY": args.judge_api_key,
    }
    overrides = {name: value for name, value in cli_env.items() if value is not None}
    if args.pool_files:
        overrides["POOL_FILES"] = ",".join(args.pool_files)

    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        runner.main()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
