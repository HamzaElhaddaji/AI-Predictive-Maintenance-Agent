import argparse
import json
from pathlib import Path

from src.maintenance.cleaner import Cleaner, inject_mess, read_data
from src.maintenance.knowledge import KnowledgeBase
from src.maintenance.model import DEFAULT_BUNDLE, ROOT, load_bundle, train
from src.maintenance.workflow import build_maintenance_agent


def main():
    parser = argparse.ArgumentParser(description="Predictive maintenance toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=ROOT / "data/train.txt")
    training.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    training.add_argument("--horizon", type=int, default=30)
    report = sub.add_parser("report")
    report.add_argument("--data", type=Path, default=ROOT / "data/test.txt")
    report.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    report.add_argument("--question", default="Full maintenance report")
    report.add_argument("--unit-id", type=int)
    report.add_argument("--output", type=Path, default=ROOT / "reports/fleet.json")
    for name in ("ingest", "mess"):
        command = sub.add_parser(name)
        if name == "ingest":
            command.add_argument("--docs", type=Path, default=ROOT / "knowledge")
            command.add_argument("--store", type=Path, default=ROOT / ".qdrant")
        else:
            command.add_argument("--data", type=Path, default=ROOT / "data/train.txt")
            command.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.command == "train":
        bundle = train(args.data, args.output, args.horizon)
        print(json.dumps(bundle["metrics"]["test"], indent=2))
    elif args.command == "ingest":
        kb = KnowledgeBase(path=args.store)
        try:
            print(f"Indexed {kb.ingest(args.docs)} chunks")
        finally:
            kb.close()
    elif args.command == "mess":
        clean = read_data(args.data)
        dirty = inject_mess(clean)
        args.output.mkdir(parents=True, exist_ok=True)
        dirty.to_csv(args.output / "messy.csv", index=False)
        repaired, quality = Cleaner().fit(clean).transform(dirty)
        repaired.to_csv(args.output / "cleaned.csv", index=False)
        print(json.dumps(quality, indent=2))
    else:
        kb = KnowledgeBase()
        try:
            kb.ingest(ROOT / "knowledge")
            result = build_maintenance_agent(load_bundle(args.bundle), kb).run(
                read_data(args.data), args.question, args.unit_id
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(result["summary"])
        finally:
            kb.close()


if __name__ == "__main__":
    main()
