"""Collect explicitly selected, reviewed JSON reports without model weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--gpt2-bindings", type=Path, required=True)
    parser.add_argument("--gpt2-probe", type=Path, required=True)
    parser.add_argument("--step-runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    outputs = {}
    for name, path in (("binding-smollm", args.bindings),
                       ("binding-gpt2", args.gpt2_bindings),
                       ("gpt2-probe", args.gpt2_probe)):
        outputs[Path(f"results/followup/{name}.json")] = json.loads(path.read_text(encoding="utf-8"))
    configs = []
    for folder in args.step_runs:
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if result["experiment"] != "scripted_causal_step_memory":
            raise ValueError("Unexpected step experiment")
        config = result["config"]
        if any(c["seed"] == config["seed"] for c in configs):
            raise ValueError("Duplicate step seed")
        if any({k: v for k, v in c.items() if k != "seed"} !=
               {k: v for k, v in config.items() if k != "seed"} for c in configs):
            raise ValueError("Step runs must differ only in seed")
        configs.append(config)
        outputs[Path(f"results/step-memory/seed-{config['seed']}.json")] = result
    manifest_path = Path("results/followup/manifest.json")
    if any(p.exists() for p in [*outputs, manifest_path]):
        raise FileExistsError("Collection refuses to overwrite existing reports")
    # Check neural reports still correspond to the implementation being published.
    for result in outputs.values():
        for name, expected in result.get("source_sha256", {}).items():
            source = Path("breadcrumb_memory") / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source changed since neural evaluation: {name}")
    for path, result in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sources = [Path(p) for p in ("breadcrumb_memory/step_memory.py", "tests/test_step_memory.py",
                                "breadcrumb_memory/lm_probe.py", "scripts/collect_followup_results.py")]
    manifest = {
        "note": "Hashes at collection. Neural reports separately record evaluation source hashes; CPU hashes identify the reviewed implementation.",
        "sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*outputs, *sources]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(outputs)} reports plus manifest")


if __name__ == "__main__":
    main()
