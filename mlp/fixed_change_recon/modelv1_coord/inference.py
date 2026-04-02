import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model.model import Change3Regressor
from train import build_dataset, split_indices, BASE_R


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inference for change3 reconstruction modelv1_coord.")
    parser.add_argument("--data-path", default=str(script_dir.parent / "data_fixed" / "training_data64_fixed_3.csv"))
    parser.add_argument("--cache-path", default=str(script_dir / "cache_change3_v1_coord.npz"))
    parser.add_argument("--coords-path", default=str(script_dir.parent / "data_fixed" / "resistor_coords_bl_origin.json"))
    parser.add_argument("--ckpt", default=str(script_dir / "outputs" / "model_best.pt"))
    parser.add_argument("--std", default=str(script_dir / "outputs" / "standardization.npz"))
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--out-json", default=str(script_dir / "outputs" / "inference_samples.json"))
    args = parser.parse_args()

    x, _x_raw, y_change, y_delta, true_ids, true_vals, _ext_nodes, _ex = build_dataset(
        Path(args.data_path), Path(args.cache_path)
    )
    tr, va, te = split_indices(len(x), args.seed)

    std_pack = np.load(args.std)
    mean = std_pack["mean"]
    std = std_pack["std"]
    x_std = ((x - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Change3Regressor(in_dim=x_std.shape[1], out_dim=y_delta.shape[1]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    coords_data = json.loads(Path(args.coords_path).read_text(encoding="utf-8"))

    rng = random.Random(args.seed + 11)
    pick = rng.sample(te, k=min(args.num_samples, len(te)))

    results = []
    with torch.no_grad():
        for idx in pick:
            xb = torch.from_numpy(x_std[idx:idx + 1]).float().to(device)
            pred = model(xb).cpu().numpy()[0]

            pred_ids = np.argsort(np.abs(pred))[-3:][::-1]
            true_id_arr = np.array(true_ids[idx], dtype=np.int64)
            true_delta_all = np.array(y_delta[idx], dtype=np.float32)

            pred_top3 = [
                {
                    "id": int(rid),
                    "pred_delta": float(pred[rid]),
                    "pred_resistance": float(BASE_R + pred[rid]),
                    "coord": coords_data.get(str(int(rid))),
                }
                for rid in pred_ids
            ]
            true_top3 = [
                {
                    "id": int(rid),
                    "true_delta": float(true_vals[idx][j]),
                    "true_resistance": float(BASE_R + true_vals[idx][j]),
                    "coord": coords_data.get(str(int(rid))),
                }
                for j, rid in enumerate(true_id_arr)
            ]

            pred_gt = int((np.abs(pred) > args.threshold).sum())
            overlap = len(set(pred_ids.tolist()).intersection(set(true_id_arr.tolist())))

            print(f"\nSample index={idx}")
            print(f"Pred top3 ids: {pred_ids.tolist()} | True ids: {true_id_arr.tolist()} | overlap={overlap}/3")
            print(f"Pred |dR|>{args.threshold}: {pred_gt}")

            results.append(
                {
                    "sample_index": int(idx),
                    "pred_top3": pred_top3,
                    "true_top3": true_top3,
                    "pred_over_threshold": pred_gt,
                    "pred_delta_all": pred.tolist(),
                    "true_delta_all": true_delta_all.tolist(),
                    "pred_resistance_all": (BASE_R + pred).tolist(),
                    "true_resistance_all": (BASE_R + true_delta_all).tolist(),
                }
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved inference json to: {out_path}")


if __name__ == "__main__":
    main()

