# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _preds(tmp_path, n=12):
    p = tmp_path / "preds.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            t = -0.8 + i * (1.6 / max(1, n - 1))
            pv = t + 0.02
            w = (pv + 1.0) * 0.5
            f.write(json.dumps({"fen": "fen%d" % i, "parent": "p%d" % (i // 2), "teacher_value": t, "pred_value": pv, "teacher_wdl": [0.4, 0.4, 0.2], "pred_wdl": [w * w, 1.0 - w * w - (1.0 - w) * (1.0 - w), (1.0 - w) * (1.0 - w)]}) + "\n")
    return str(p)


def test_eval_calibration(tmp_path):
    preds = _preds(tmp_path)
    out = str(tmp_path / "cal.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "calibration", "eval_calibration.py"), "--preds", preds, "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out))
    assert "near-equal" in rep or "medium" in rep


def test_wdl_calibration(tmp_path):
    preds = _preds(tmp_path)
    out = str(tmp_path / "wdl.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "calibration", "wdl_calibration.py"), "--preds", preds, "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(out))
    assert rep["mismatch_rate"] < 0.5


def test_boundary(tmp_path):
    preds = _preds(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "calibration", "boundary_analysis.py"), "--preds", preds, "--alpha", "0.1", "--beta", "0.3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "boundary_flip_rate" in r.stdout


def test_ranking(tmp_path):
    preds = _preds(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "calibration", "ranking_error.py"), "--preds", preds], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rank_err_rate" in r.stdout


def test_search_tools(tmp_path):
    preds = _preds(tmp_path, 6)
    pos = tmp_path / "pos.txt"
    pos.write_text("fen0\nfen1\nfen2\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "search", "run_search_bench.py"), "--preds", preds, "--positions", str(pos), "--depth", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "search", "run_search_bench.py"), "--preds", preds, "--positions", str(pos), "--depth", "1", "--out", str(a)], capture_output=True, text=True)
    assert r.returncode == 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "search", "run_search_bench.py"), "--preds", preds, "--positions", str(pos), "--depth", "1", "--out", str(b)], capture_output=True, text=True)
    assert r.returncode == 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "search", "search_diff.py"), "--a", str(a), "--b", str(b)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
