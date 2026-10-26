#!/usr/bin/env python
"""Prepare the next 24h segment in place (hangang port of hawk chain2/prepare_segment.sh).

  TRAIN_SRC=<dir of train .extxyz> python prepare_segment.py [--dry-run]

1) progress: the last `train_data/<link>` named in the newest loss_train*.out is where the
   previous segment stopped; it is redone (it may be half trained).
2) checkpoint: the newest *loadable* of model_train.pkl / fname_pkl is promoted to
   fname_pkl (previous one kept as .prev) and restart is set to true.
3) train_data/ is rebuilt as <N>_<shard> links (N zero-padded to the shard count, since
   DensTrainer visits them in name order), continuing the order the trainer actually used
   from that shard on, so a restart neither retrains the front shards nor skips any.
4) a segment that made no progress bumps a fail counter; 2 in a row -> STOP file.
State lives in segment_state.json ({"seg", "fails", "first"}).
"""
import glob, json, os, re, shutil, sys

SRC = os.environ["TRAIN_SRC"]  # set by segment.sh
MAX_FAILS = 2
dry = "--dry-run" in sys.argv


def load_state():
    try:
        return json.load(open("segment_state.json"))
    except FileNotFoundError:
        return {"seg": 0, "fails": 0}


def current_order():
    """(link names in trainer order, shard names in SRC) for the current train_data."""
    if os.path.isdir("train_data") and not os.path.islink("train_data"):  # built by us
        links = sorted(f for f in os.listdir("train_data") if f.endswith(".extxyz"))
        return links, [re.sub(r"^\d+_", "", l, count=1) for l in links]
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".extxyz"))  # plain dir / first run
    return files, files


def last_started(links):
    """(log, position in links) of the last train shard started, or (log, None)."""
    logs = sorted(glob.glob("loss_train*.out"), key=os.path.getmtime)
    if not logs:
        return None, None
    hits = re.findall(r"from train_data/(\S+\.extxyz)", open(logs[-1]).read())
    return logs[-1], (links.index(hits[-1]) if hits and hits[-1] in links else None)


def pick_checkpoint(target):
    import torch
    best = None
    for name in ("model_train.pkl", target):
        if not os.path.exists(name):
            continue
        try:
            c = torch.load(name, map_location="cpu", weights_only=False)
            for k in ("params", "loss"):
                c[k]
            if best is None or os.path.getmtime(name) > best[0]:
                best = (os.path.getmtime(name), name, c["loss"].get("epoch"))
        except Exception as e:
            print(f"  {name}: unreadable ({type(e).__name__}: {e})")
    return best


def main():
    if os.path.exists("STOP"):
        sys.exit("STOP file present")
    st = load_state()
    links, names = current_order()
    n = len(names)
    missing = [x for x in set(names) if not os.path.exists(os.path.join(SRC, x))]
    if missing:
        sys.exit(f"{len(missing)} shards of train_data not in TRAIN_SRC={SRC}, e.g. {missing[0]}")
    cfg = json.load(open("input.json"))
    target = cfg["NN"]["fname_pkl"]

    log, pos = last_started(links)
    if log is None:
        pos = 0
        print("1) no log yet: first segment")
    elif pos is None:
        pos = 0
        st["fails"] += 1
        print(f"1) {log}: no train shard started -> no progress (fails={st['fails']})")
    else:
        st["fails"] = 0
        print(f"1) {log}: last started {links[pos]} (position {pos} of {n}) -> redo it")
    if st["fails"] >= MAX_FAILS:
        if not dry:
            open("STOP", "w").write(f"{st['fails']} segments in a row without progress\n")
        sys.exit("too many segments without progress: wrote STOP")
    order = names[pos:] + names[:pos]
    print(f"   next order starts at {order[0]} ({n} shards)")

    ck = pick_checkpoint(target) if log is not None else None
    if log is not None and ck is None:
        sys.exit("2) a previous segment ran but no loadable checkpoint: refusing to restart from scratch")
    print(f"2) newest loadable checkpoint: {ck[1]} (epoch {ck[2]})" if ck else "2) no checkpoint: start from scratch")

    if dry:
        print("dry run: nothing changed")
        return
    if ck and ck[1] != target:
        if os.path.exists(target):
            shutil.copy2(target, target + ".prev")
        shutil.copy2(ck[1], target)
    cfg["NN"]["restart"] = bool(ck)
    json.dump(cfg, open("input.json.tmp", "w"), indent=2)
    os.replace("input.json.tmp", "input.json")

    width = len(str(n - 1))
    shutil.rmtree("train_data.new", ignore_errors=True)
    os.mkdir("train_data.new")
    for i, name in enumerate(order):
        os.symlink(os.path.join(SRC, name), f"train_data.new/{i:0{width}d}_{name}")
    if os.path.islink("train_data"):
        os.unlink("train_data")
    elif os.path.isdir("train_data"):
        shutil.rmtree("train_data")
    os.rename("train_data.new", "train_data")

    st.update(seg=st["seg"] + 1, first=order[0])
    st.pop("offset", None)
    json.dump(st, open("segment_state.json", "w"))
    print(f"3) train_data: {n} links starting at {order[0]}; restart={cfg['NN']['restart']}; segment {st['seg']}")


if __name__ == "__main__":
    main()
