import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_PATH = "../../data/training_data64Nodes_2.csv"
DEFAULT_LEVELS = "clean=0,40dB=0.01,30dB=0.0316227766,20dB=0.1"


def sanitize_name(raw_name):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_name.strip())
    safe = safe.strip("._-")
    return safe or "item"


def validate_tag_value(raw_tag, option_name):
    if not raw_tag or not raw_tag.strip():
        raise SystemExit(f"{option_name} requires a non-empty value.")
    if any(ch in raw_tag for ch in "$ {}"):
        raise SystemExit(
            f"{option_name} looks like an unexpanded shell placeholder: {raw_tag!r}. "
            "Please pass the real run tag, not ${TAG}."
        )


def resolve_path(raw_path, script_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    workspace_root = script_dir.parents[1]
    gnn_root = workspace_root / "gnn"
    candidates = [path, script_dir / path, gnn_root / path, workspace_root / path]
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    return (workspace_root / path).resolve()


def parse_level_specs(raw_value):
    specs = []
    for chunk in str(raw_value).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid noise level item: {item!r}. Use LABEL=STD, for example 40dB=0.01.")
        label, std_text = item.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"Invalid noise level item: {item!r}. Label cannot be empty.")
        specs.append((label, float(std_text.strip())))
    if not specs:
        raise SystemExit("At least one noise level must be provided.")
    return specs


def build_single_model_command(python_bin, script_path, data_path, run_tag, noise_std, noise_seed):
    cmd = [str(python_bin), str(script_path), "--data-path", str(data_path), "--dataset-tag", run_tag]
    if noise_std > 0:
        cmd.extend(["--noise-std", str(noise_std), "--noise-seed", str(noise_seed)])
    return cmd


def build_joint_command(python_bin, script_path, data_path, run_tag, cls_dir, reg_dir, noise_std, noise_seed, out_dir):
    cmd = [
        str(python_bin),
        str(script_path),
        "--data-path",
        str(data_path),
        "--dataset-tag",
        run_tag,
        "--cls-dir",
        str(cls_dir),
        "--reg-dir",
        str(reg_dir),
        "--out-dir",
        str(out_dir),
    ]
    if noise_std > 0:
        cmd.extend(["--noise-std", str(noise_std), "--noise-seed", str(noise_seed)])
    return cmd


def copy_if_exists(src_path, dst_path):
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    return False


def metrics_copy_name(noisy, label):
    if noisy:
        return f"noise_eval_{sanitize_name(label)}.json"
    return f"inference_eval_{sanitize_name(label)}.json"


def samples_copy_name(label):
    return f"inference_samples_{sanitize_name(label)}.json"


def format_command(cmd):
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def main():
    parser = argparse.ArgumentParser(description="Run a full clean/40dB/30dB/20dB noise evaluation suite for a trained GNN_NOISE run.")
    parser.add_argument("--run-tag", required=True, help="Training run tag under outputs/, for example training_data64Nodes_2_noiseft_struct_boundary_v2_20260402.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--cls-dir", default="GNN_NOISE/CLS_modelo3_ft_v2")
    parser.add_argument("--reg-dir", default="GNN_NOISE/REG_o4a2_ft_v2")
    parser.add_argument("--joint-script", default="gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py")
    parser.add_argument("--levels", default=DEFAULT_LEVELS, help="Comma-separated LABEL=STD list, for example clean=0,40dB=0.01,30dB=0.0316227766,20dB=0.1")
    parser.add_argument("--noise-seed", type=int, default=20260402)
    parser.add_argument("--joint-prefix", default="gnn_cmei")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parents[1]
    validate_tag_value(args.run_tag, "--run-tag")
    run_tag = sanitize_name(args.run_tag)
    data_path = resolve_path(args.data_path, script_dir)
    cls_dir = resolve_path(args.cls_dir, script_dir)
    reg_dir = resolve_path(args.reg_dir, script_dir)
    joint_script = resolve_path(args.joint_script, script_dir)
    cls_script = cls_dir / "inference.py"
    reg_script = reg_dir / "inference.py"
    levels = parse_level_specs(args.levels)

    if not cls_script.exists():
        raise SystemExit(f"Missing CLS inference script: {cls_script}")
    if not reg_script.exists():
        raise SystemExit(f"Missing REG inference script: {reg_script}")
    if not joint_script.exists():
        raise SystemExit(f"Missing joint inference script: {joint_script}")

    print(f"[Suite] run_tag={run_tag}")
    print(f"[Suite] data_path={data_path}")

    summary_rows = []
    for label, noise_std in levels:
        label_safe = sanitize_name(label)
        joint_out_dir = Path("outputs") / f"{sanitize_name(args.joint_prefix)}_{run_tag}_{label_safe}"
        cls_cmd = build_single_model_command(args.python_bin, cls_script, data_path, run_tag, noise_std, args.noise_seed)
        reg_cmd = build_single_model_command(args.python_bin, reg_script, data_path, run_tag, noise_std, args.noise_seed)
        joint_cmd = build_joint_command(
            args.python_bin,
            joint_script,
            data_path,
            run_tag,
            cls_dir,
            reg_dir,
            noise_std,
            args.noise_seed,
            joint_out_dir,
        )

        print(f"\n[Level {label}]")
        print(format_command(cls_cmd))
        print(format_command(reg_cmd))
        print(format_command(joint_cmd))

        if args.dry_run:
            continue

        subprocess.run(cls_cmd, cwd=workspace_root, check=True)
        subprocess.run(reg_cmd, cwd=workspace_root, check=True)
        subprocess.run(joint_cmd, cwd=workspace_root, check=True)

        noisy = noise_std > 0
        cls_out_dir = cls_dir / "outputs" / run_tag
        reg_out_dir = reg_dir / "outputs" / run_tag
        joint_metrics_path = workspace_root / "gnn" / "GNN_CMEI_INFERENCE" / joint_out_dir / run_tag / "cmei_metrics.json"

        cls_metrics_src = cls_out_dir / ("noise_eval.json" if noisy else "inference_eval.json")
        reg_metrics_src = reg_out_dir / ("noise_eval.json" if noisy else "inference_eval.json")
        cls_samples_src = cls_out_dir / "inference_samples.json"
        reg_samples_src = reg_out_dir / "inference_samples.json"

        copy_if_exists(cls_metrics_src, cls_out_dir / metrics_copy_name(noisy, label))
        copy_if_exists(reg_metrics_src, reg_out_dir / metrics_copy_name(noisy, label))
        copy_if_exists(cls_samples_src, cls_out_dir / samples_copy_name(label))
        copy_if_exists(reg_samples_src, reg_out_dir / samples_copy_name(label))

        row = {"label": label, "noise_std": noise_std}
        if cls_metrics_src.exists():
            cls_metrics = json.loads(cls_metrics_src.read_text(encoding="utf-8"))
            row["cls_macro_f1"] = cls_metrics.get("test_macro_f1")
        if reg_metrics_src.exists():
            reg_metrics = json.loads(reg_metrics_src.read_text(encoding="utf-8"))
            row["reg_mae_all"] = reg_metrics.get("mae_all")
            row["reg_mae_changed"] = reg_metrics.get("mae_changed")
            row["reg_count_macro_f1"] = reg_metrics.get("count_macro_f1")
        if joint_metrics_path.exists():
            joint_metrics = json.loads(joint_metrics_path.read_text(encoding="utf-8"))
            row["joint_cmei"] = joint_metrics.get("scores", {}).get("CMEI")
            row["joint_macro_f1"] = joint_metrics.get("macro_f1")
            row["joint_id_recall"] = joint_metrics.get("id_recall")
            row["joint_mse_all_edges"] = joint_metrics.get("mse_all_edges")
        summary_rows.append(row)

    if args.dry_run:
        print("\n[Dry Run] No commands were executed.")
        return

    print("\n[Suite Summary]")
    for row in summary_rows:
        text = [f"label={row['label']}", f"noise_std={row['noise_std']:.6f}"]
        if "cls_macro_f1" in row:
            text.append(f"cls_macro_f1={row['cls_macro_f1']:.4f}")
        if "reg_mae_all" in row:
            text.append(f"reg_mae_all={row['reg_mae_all']:.4f}")
        if "reg_mae_changed" in row:
            text.append(f"reg_mae_changed={row['reg_mae_changed']:.4f}")
        if "joint_cmei" in row:
            text.append(f"joint_CMEI={row['joint_cmei']:.2f}")
        print(" | ".join(text))


if __name__ == "__main__":
    main()
