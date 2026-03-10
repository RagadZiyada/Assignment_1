from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time


def run_step(script_path: Path, args_list: list[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args_list
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parent.parent

    step1_dir = output_dir / "01_prepared"
    step2_dir = output_dir / "02_features"
    step3_dir = output_dir / "03_filtered"
    step4_dir = output_dir / "04_ga"
    step5_dir = output_dir / "05_model"

    run_step(base_dir / "src" / "prepare_data.py", [
        "--data_dir", str(data_dir),
        "--output_dir", str(step1_dir)
    ])

    run_step(base_dir / "src" / "extract_features.py", [
        "--input_dir", str(step1_dir),
        "--output_dir", str(step2_dir)
    ])

    run_step(base_dir / "src" / "filter_select.py", [
        "--input_dir", str(step2_dir),
        "--output_dir", str(step3_dir)
    ])

    run_step(base_dir / "src" / "ga_select.py", [
        "--input_dir", str(step3_dir),
        "--output_dir", str(step4_dir)
    ])

    run_step(base_dir / "src" / "train_model.py", [
        "--input_dir", str(step4_dir),
        "--output_dir", str(step5_dir)
    ])

    for file_path in [
        step4_dir / "selected_features.csv",
        step5_dir / "metrics.json",
        step5_dir / "predictions.csv",
        step5_dir / "model.joblib"
    ]:
        if file_path.exists():
            shutil.copy2(file_path, output_dir / file_path.name)

    total_runtime = round(time.time() - start_time, 2)

    with open(output_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"status": "completed", "runtime_seconds": total_runtime}, f, indent=2)

    print("Pipeline completed")
    print("Runtime (seconds):", total_runtime)


if __name__ == "__main__":
    main()