import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model.model import Change3Regressor
from train import build_dataset, split_indices


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inference for change3 reconstruction modelv1_new.")
    parser.add_argument("--data-path", default=str(script_dir.parent / "data_fixed" / "training_data64_fixed_3.csv"))
    parser.add_argument("--cache-path", default=str(script_dir / "cache_change3_v1_new.npz"))
    parser.add_argument("--ckpt", default=str(script_dir / "outputs" / "model_best.pt"))
    parser.add_argument("--std", default=str(script_dir / "outputs" / "standardization.npz"))
    parser.add_argument("--seed", type=int, default=20260322)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out-json", default=str(script_dir / "outputs" / "inference_samples.json"))
    args = parser.parse_args()

    x, _x_raw, _y_change, y_delta, true_ids, _true_vals, _ext_nodes, _ex = build_dataset(
        Path(args.data_path), Path(args.cache_path)
    )
    _tr, _va, te = split_indices(len(x), args.seed)

    std_pack = np.load(args.std)
    mean = std_pack["mean"]
    std = std_pack["std"]
    x_std = ((x - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Change3Regressor(in_dim=x_std.shape[1], out_dim=y_delta.shape[1]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rng = random.Random(args.seed + 17)
    pick = rng.sample(te, k=min(args.num_samples, len(te)))

    results = []
    with torch.no_grad():
        for idx in pick:
            xb = torch.from_numpy(x_std[idx:idx + 1]).float().to(device)
            pred = model(xb).cpu().numpy()[0]

            pred_ids = np.argsort(np.abs(pred))[-3:][::-1]
            true_id_arr = np.array(true_ids[idx], dtype=np.int64)

            pred_id_delta = [float(pred[rid]) for rid in pred_ids]
            true_id_delta = [float(y_delta[idx][rid]) for rid in true_id_arr]

            results.append(
                {
                    "pred_id": pred_ids.astype(int).tolist(),
                    "true_id": true_id_arr.astype(int).tolist(),
                    "pred_id_delta": pred_id_delta,
                    "true_id_delta": true_id_delta,
                    "pred_delta_all": pred.tolist(),
                }
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved inference json to: {out_path}")


if __name__ == "__main__":
    main()

