"""Materialize the pre-declared token slice (docs/01 C0.1).

Fetches rows of the Pythia preshuffled stream by HTTP range request, verifies them by
decoding, writes a uint16 array plus a sha256 and the config that produced it.

    python experiments/phase0/slice.py experiments/phase0/configs/slice.yaml out/slice
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import requests
import yaml


def _cfg(path):
    c = yaml.safe_load(open(path))
    # yaml coerces carelessly; validate what we use
    for k in ("first_step", "num_steps"):
        assert isinstance(c["anchor"][k], int), k
    for k in ("row_tokens", "rows_per_step", "bin_bytes"):
        assert isinstance(c["source"][k], int), k
    return c


def fetch_rows(src, row0, nrows, session=None):
    """Rows [row0, row0+nrows) of the preshuffled stream as (nrows, row_tokens) uint16."""
    s = session or requests.Session()
    rt, nb = src["row_tokens"], src["bin_bytes"]
    byte0, byte1 = row0 * rt * 2, (row0 + nrows) * rt * 2  # [byte0, byte1)
    out = bytearray()
    pos = byte0
    while pos < byte1:
        i, off = divmod(pos, nb)
        end = min(byte1, (i + 1) * nb)
        url = (f"https://huggingface.co/datasets/{src['repo']}/resolve/main/"
               + src["bin_pattern"].format(i=i))
        r = s.get(url, headers={"Range": f"bytes={off}-{end - i * nb - 1}"}, timeout=120)
        assert r.status_code == 206, (r.status_code, url)
        assert len(r.content) == end - pos, (len(r.content), end - pos)
        out += r.content
        pos = end
    return np.frombuffer(bytes(out), dtype=np.uint16).reshape(nrows, rt)


def main(cfg_path, out_dir):
    c = _cfg(cfg_path)
    src, a = c["source"], c["anchor"]
    row0 = a["first_step"] * src["rows_per_step"]
    nrows = a["num_steps"] * src["rows_per_step"]
    rows = fetch_rows(src, row0, nrows)
    assert rows.max() < 50304, int(rows.max())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "anchor_rows.npy", rows)
    sha = hashlib.sha256(rows.tobytes()).hexdigest()
    meta = {"config": c, "row0": row0, "nrows": nrows, "tokens": int(rows.size),
            "sha256": sha}
    json.dump(meta, open(out / "anchor_rows.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
