"""
Merge an Azure fine-tuning result file into sft_loss_curves_clean.json.

Usage:
    python merge_loss_curve.py <config_name> <file_id>

Example:
    python merge_loss_curve.py sft_c2 file-xxxx
"""
import csv, io, json, os, sys
from openai import AzureOpenAI

CLEAN_PATH = "data/results/sft_loss_curves_clean.json"

client = AzureOpenAI(
    api_key="",
    azure_endpoint="https://cis-5270-team-10.openai.azure.com",
    api_version="2025-04-01-preview",
)

def safe_float(s):
    try: return float(s)
    except: return None

def merge(name, file_id):
    content = client.files.content(file_id).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(content)))

    curves = {}
    if os.path.exists(CLEAN_PATH):
        with open(CLEAN_PATH) as f:
            curves = json.load(f)

    curves[name] = {
        "steps":      [int(r["step"]) for r in rows],
        "train_loss": [safe_float(r["train_loss"]) for r in rows],
        "valid_loss": [safe_float(r["valid_loss"]) for r in rows],
    }

    with open(CLEAN_PATH, "w") as f:
        json.dump(curves, f, indent=2)

    train_vals = [v for v in curves[name]["train_loss"] if v is not None]
    print(f"merged '{name}': {len(rows)} steps, final train_loss={train_vals[-1]:.4f}")
    print(f"saved to {CLEAN_PATH}  (keys: {list(curves.keys())})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python merge_loss_curve.py <name> <file_id>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
