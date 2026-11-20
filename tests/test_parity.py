"""Train/inference parity checks (CPU, a few seconds). Run: python tests/test_parity.py

For every model variant the trainer can produce, the step-wise kernel that ships in a submission (wnn/infer.py
StepModel, numba and numpy paths) must reproduce GRUModel.forward in fp32, including non-finite inputs.
Also builds a throw-away submission in the flat zip layout and runs its solution.py.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wnn.features import n_bank, parse_bank  # noqa: E402
from wnn.infer import HAVE_NUMBA, StepModel, export_npz  # noqa: E402
from wnn.model import GRUModel  # noqa: E402

TOL = 2e-5
VARIANTS = {  # name -> GRUModel kwargs (+ bank spec)
    "gru128": dict(hidden=128),
    "gru64x2": dict(hidden=64, layers=2),
    "proj64": dict(hidden=64, in_proj=64),
    "highway2": dict(hidden=64, head="highway", head_layers=2),
    "chrono": dict(hidden=64, chrono=3000.0),
    "bank": dict(hidden=32, bank="10,100"),
    "bank_scaled": dict(hidden=32, bank="10,100:s"),
}


def build(name, tmp, seed=0):
    kw = dict(VARIANTS[name]); bank = kw.pop("bank", "")
    torch.manual_seed(seed)
    n_in = n_bank(112, len(parse_bank(bank)[0])) if bank else 112
    m = GRUModel(n_in, **kw).eval()
    with torch.no_grad():  # non-trivial normalisation and LayerNorm parameters
        m.mean.copy_(torch.randn(n_in) * 0.5); m.std.copy_(torch.rand(n_in) + 0.5)
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    path = str(Path(tmp) / f"{name}.npz")
    export_npz(m, path, bank=bank)
    return m, path, bank


def torch_ref(m, X, bank):
    from wnn.data import maybe_augment

    Xa = maybe_augment(X, False, bank)
    with torch.no_grad():
        return m(torch.from_numpy(Xa)[None])[0][0].numpy()


def check_variants(T=1500):
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for name in VARIANTS:
            if name == "bank_scaled" and not HAVE_NUMBA:
                # x - EMA is divided by a running scale that starts at 0 (+1e-6): in the first steps the ratio is
                # rounding noise (up to ~0.1), so numpy and the fastmath kernel legitimately differ.
                continue
            m, path, bank = build(name, tmp)
            X = (rng.normal(size=(T, 112)) * 2).astype(np.float32)
            if not bank:  # non-finite inputs (the bank's EMA state would propagate them by design)
                X[5, 3] = np.nan; X[7, 5] = np.inf; X[9, 7] = -np.inf; X[11] = np.nan
            ref = torch_ref(m, X, bank)
            sm = StepModel(path)
            out = np.array([sm.step(x).copy() for x in X])
            sm.reset()
            out2 = np.array([sm.step(x).copy() for x in X[:50]])  # reset() must restore the initial state exactly
            assert np.isfinite(out).all(), name
            d, d2 = np.abs(out - ref).max(), np.abs(out2 - ref[:50]).max()
            # the numpy bank kernel is documented to differ at step 0 only
            if bank and not HAVE_NUMBA:
                d, d2 = np.abs(out[1:] - ref[1:]).max(), np.abs(out2[1:] - ref[1:50]).max()
            print(f"  {name:12s} max|step - torch| = {d:.1e} (after reset {d2:.1e})")
            assert d < TOL and d2 < TOL, (name, d, d2)


def check_submission_layout():
    """solution.py + infer.py + features.py side by side (the zip layout), 2 models, scorer-style calls."""
    import shutil
    from dataclasses import dataclass

    @dataclass
    class DP:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "sub"; stage.mkdir()
        shutil.copy(ROOT / "submit" / "solution.py", stage / "solution.py")
        for fn in ("infer.py", "features.py"):
            shutil.copy(ROOT / "wnn" / fn, stage / fn)
        ms = []
        for i, name in enumerate(("gru128", "bank")):  # the bank model exercises the flat-layout features import
            m, path, bank = build(name, tmp, seed=i)
            shutil.copy(path, stage / f"weights_{i}.npz"); ms.append((m, bank))
        code = (
            "import sys, numpy as np; sys.path.insert(0, r'%s'); import solution as s\n"
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class DP:\n    seq_ix: int\n    step_in_seq: int\n    need_prediction: bool\n    state: np.ndarray\n"
            "X = np.load(r'%s'); pm = s.PredictionModel(); out = []\n"
            "for seq in (7, 3):\n"
            "    for t in range(len(X)):\n"
            "        p = pm.predict(DP(seq, t, t >= 99, X[t]))\n"
            "        assert (p is None) == (t < 99)\n"
            "        out.append(np.zeros(2, np.float32) if p is None else p)\n"
            "        assert out[-1].dtype == np.float32 and out[-1].shape == (2,)\n"
            "np.save(r'%s', np.array(out)); print(pm.backend_name)\n"
        ) % (stage, Path(tmp) / "X.npy", Path(tmp) / "out.npy")
        X = (np.random.default_rng(1).normal(size=(400, 112)) * 2).astype(np.float32)
        np.save(Path(tmp) / "X.npy", X)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
        assert r.returncode == 0, r.stderr[-2000:]
        out = np.load(Path(tmp) / "out.npy")
        ref = np.mean([torch_ref(m, X, bank) for m, bank in ms], 0)
        ref[:99] = 0
        ref = np.concatenate([ref, ref])  # second sequence must restart from a clean state
        d = np.abs(out - ref).max()
        print(f"  flat-layout solution.py [{r.stdout.strip()}] 2-model mean vs torch: {d:.1e}")
        assert d < TOL * 5 if not HAVE_NUMBA else d < TOL, d


if __name__ == "__main__":
    print(f"backend: {'numba' if HAVE_NUMBA else 'numpy'}")
    check_variants()
    check_submission_layout()
    if HAVE_NUMBA and not os.environ.get("WNN_PARITY_CHILD"):  # repeat everything on the numpy fallback path
        r = subprocess.run([sys.executable, __file__], env=dict(os.environ, WNN_NO_NUMBA="1", WNN_PARITY_CHILD="1"))
        assert r.returncode == 0
    print("parity OK")
