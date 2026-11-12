"""Runs: tahap + keputusan -> konfigurasi dan run id"""
from common import read_json
from config import (BACKBONES, BASE, LANES, PROXIES, RESULTS, RUNS, SEEDS, SMOKE_RESULTS, SMOKE_RUNS,
                    ZEROSHOT_BACKBONE)

ZEROSHOT_ID = "a_zeroshot-flan"
BASELINES = ["base_lai", "base_mfr-mln", "base_mfr-all", "base_rule"]
LR_NAMES = {"lr3e-4": 3e-4, "lr3e-3": 3e-3}


def results_dir(smoke):
    return SMOKE_RESULTS if smoke else RESULTS


def run_dir(run_id, smoke=False):
    return (SMOKE_RUNS if smoke else RUNS) / run_id


def a_config(backbone):
    return {"winner": backbone, "prefix": f"a_{backbone}", "params": {**BASE, "backbone": backbone}}


def stage_b_base(win):
    """pemenang A -> titik awal tahap B (proxy jika pemenang punya proxy)"""
    winner = need(win, "a")["params"]["backbone"]
    return a_config(PROXIES.get(winner, winner))


def load_winners(smoke=False):
    """decision.json per tahap -> dict"""
    win = {}
    for stage in "abcdef":
        path = results_dir(smoke) / f"stage_{stage}" / "decision.json"
        if path.exists():
            win[stage] = read_json(path)
    # pemenang sementara untuk smoke test
    if smoke:
        win.setdefault("a", a_config("t5-small"))
        for stage in "bcd":
            win.setdefault(stage, stage_b_base(win))
    return win


def need(win, stage):
    if stage not in win:
        raise RuntimeError(f"keputusan tahap {stage} belum ada, jalankan eval tahap {stage} dulu")
    return win[stage]


def stage_configs(stage, win=None):
    """tahap -> list config (name, prefix, params, cost)"""
    win = load_winners() if win is None else win
    if stage == "a":
        return [{"name": bb, "prefix": f"a_{bb}", "params": {**BASE, "backbone": bb}, "cost": info["cost"]}
                for bb, info in BACKBONES.items()]
    if stage == "b":
        prev = stage_b_base(win)
        out = []
        for cost, (mix, up) in enumerate([("a", 1), ("b", 1), ("c", 1), ("b", 4), ("c", 4)], 1):
            name = f"{mix}{up}"
            prefix = prev["prefix"] if (mix, up) == (BASE["mix"], BASE["up"]) else f"b_{name}"
            out.append({"name": name, "prefix": prefix, "params": {**prev["params"], "mix": mix, "up": up},
                        "cost": cost})
        return out
    if stage == "c":
        prev = need(win, "b")
        return [{"name": "delim", "prefix": prev["prefix"], "params": prev["params"], "cost": 1},
                {"name": "plain", "prefix": "c_plain", "params": {**prev["params"], "fmt": "plain"}, "cost": 2}]
    if stage == "d":
        prev = need(win, "c")
        out = [{"name": "lr1e-3", "prefix": prev["prefix"], "params": prev["params"], "cost": 1}]
        for name, lr in LR_NAMES.items():
            out.append({"name": name, "prefix": f"d_{name}", "params": {**prev["params"], "lr": lr}, "cost": 2})
        return out
    if stage == "e":
        prev = need(win, "d")
        target = need(win, "a")["params"]["backbone"]
        out = [{"name": "d-final", "prefix": prev["prefix"], "params": prev["params"], "cost": 1}]
        # tanpa proxy: pemenang D langsung jadi kandidat final
        if prev["params"]["backbone"] == target:
            return out
        base, final = a_config(target)["params"], {**prev["params"], "backbone": target}
        out.append({"name": "target-base", "prefix": f"a_{target}", "params": base, "cost": 2})
        # config terpilih sama dengan config dasar -> tidak ada run baru
        if final != base:
            out.append({"name": "target-final", "prefix": f"e_{target}", "params": final, "cost": 3})
        return out
    raise ValueError(stage)


def run_ids(config):
    return [f"{config['prefix']}_s{seed}" for seed in SEEDS]


def get_run(run_id, smoke=False):
    """run id -> spesifikasi run"""
    if run_id == ZEROSHOT_ID:
        return {"id": run_id, "kind": "zeroshot", "backbone": ZEROSHOT_BACKBONE, "fmt": "plain"}
    if run_id in BASELINES:
        return {"id": run_id, "kind": "baseline", "fmt": "delim"}
    prefix, seed = run_id.rsplit("_s", 1)
    win = load_winners(smoke)
    for config in stage_configs(run_id[0], win):
        if config["prefix"] == prefix:
            return {"id": run_id, "kind": "train", "stage": run_id[0], "config": config["name"],
                    **config["params"], "seed": int(seed), "lane": LANES[int(seed)]}
    raise ValueError(f"run id tidak dikenal: {run_id}")


def train_runs(stage, win=None):
    """tahap -> run id yang perlu dilatih di tahap itu"""
    return [rid for c in stage_configs(stage, win) if c["prefix"].startswith(stage) for rid in run_ids(c)]
