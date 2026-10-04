import argparse
import json
from dataclasses import fields
from pathlib import Path

from .experiment import Config, hardware, run


def main():
    parser = argparse.ArgumentParser(description="Bounded synthetic memory lab. No LLM downloads or network calls.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect PyTorch/GPU and perform a forward/backward kernel test")
    train = sub.add_parser("train", help="Train teacher/student and evaluate all controls")
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--seed", type=int)
    train.add_argument("--steps", type=int, help="Override student steps")
    train.add_argument("--max-minutes", type=float)
    manual = sub.add_parser("demo", help="Ask about a generated or manually edited synthetic history")
    manual.add_argument("--checkpoint", type=Path, required=True)
    manual.add_argument("--history", type=Path, help="JSON with records and query; see examples/history.json")
    manual.add_argument("--length", type=int, default=40)
    manual.add_argument("--seed", type=int, default=99)
    manual.add_argument("--query-position", type=int, default=0)
    manual.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(hardware(), indent=2))
        return
    if args.command == "demo":
        from .demo import demo
        demo(args.checkpoint, args.history, args.length, args.seed, args.query_position, args.device)
        return
    data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    for arg, key in ((args.seed, "seed"), (args.steps, "student_steps"), (args.max_minutes, "max_minutes")):
        if arg is not None:
            data[key] = arg
    run(Config(**data), args.output, args.device)


if __name__ == "__main__":
    main()
