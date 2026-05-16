#!/usr/bin/env python3
"""Wave6 analyzer: e10 top-K retrieval (content-vs-capacity discriminator)
plus e13 multi-task retry. Reads cell JSONs from
v2/experiments/e10_topk_retrieval/ and e13_multi_task_capability/ and
prints the decisive contrasts."""
import json, sys, glob, os

ROOT = "/Users/gabiri/projects/RCV-HC/v2/experiments"


def load_e10():
    rows = []
    for path in sorted(glob.glob(f"{ROOT}/e10_topk_retrieval/*.json")):
        try:
            j = json.load(open(path))
        except Exception as exc:
            print(f"# skip {path}: {exc}", file=sys.stderr)
            continue
        variant = j.get("variant") or j.get("config", {}).get("variant")
        after = j.get("after") or {}
        before = j.get("before") or {}
        base = after.get("base") if after.get("base") is not None else before.get("base")
        post_real = after.get("real")
        post_rand = after.get("rand")
        post_zero = after.get("zero")
        post_off  = after.get("off")
        if base is None or post_real is None:
            print(f"# {os.path.basename(path)}: missing nll keys -> {list(j.keys())}",
                  file=sys.stderr)
            continue
        rows.append({
            "variant": variant or os.path.basename(path).replace(".json", ""),
            "base": base,
            "post": post_real,
            "post_rand": post_rand,
            "post_zero": post_zero,
            "post_off": post_off,
            "delta": post_real - base,
            "K": j.get("K"),
        })
    return rows


def load_e13():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/e13_multi_task_capability/*.json")):
        try:
            j = json.load(open(path))
        except Exception:
            continue
        out.append((os.path.basename(path), j))
    return out


def main():
    print("# Wave6 results — e10 top-K + e13 multi-task retry\n")
    print("## A. e10 — content-vs-capacity discriminator\n")
    rows = load_e10()
    if not rows:
        print("(no e10 results yet)\n")
    else:
        print("| variant | K | base | post_real | post_rand | post_zero | post_off | Δ (post_real-base) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            def fmt(v): return f"{v:.3f}" if isinstance(v,(int,float)) else "—"
            print(f"| {r['variant']} | {r.get('K') or '—'} | {fmt(r['base'])} | {fmt(r['post'])} | {fmt(r['post_rand'])} | {fmt(r['post_zero'])} | {fmt(r['post_off'])} | {r['delta']:+.3f} |")
        print()
        # Decisive contrast: real top-K vs random top-K
        by_v = {r["variant"]: r for r in rows}
        real_k8 = by_v.get("topk_cosine_real_K8")
        rand_k8 = by_v.get("topk_cosine_random_K8")
        all_real = by_v.get("all_attend_real")
        all_rand = by_v.get("all_attend_random_renorm15")
        print("## B. Decisive contrasts\n")
        if real_k8 and rand_k8:
            gap = rand_k8["post"] - real_k8["post"]
            print(f"- top-K=8 real vs random: post_real={real_k8['post']:.3f} "
                  f"vs random={rand_k8['post']:.3f} → gap={gap:+.3f}")
            if gap >= 1.0:
                print("  → **content-sensitive** (real beats random at K=8)")
            else:
                print("  → **content-insensitive** (no real-vs-random gap at K=8)")
        if all_real and all_rand:
            gap = all_rand["post"] - all_real["post"]
            print(f"- all-attend real vs random_renorm15: gap={gap:+.3f}")
        if real_k8 and all_real:
            gap = real_k8["post"] - all_real["post"]
            print(f"- top-K=8 real vs all-attend real: post diff={gap:+.3f} "
                  f"(near 0 → top-K is as good as all-attend; "
                  ">> 0 → top-K loses information)")

    print("\n## C. e13 multi-task retry\n")
    e13 = load_e13()
    if not e13:
        print("(no e13 results yet)\n")
    else:
        for name, j in e13:
            print(f"### {name}")
            for task in ("wikitext2", "lambada", "hellaswag", "gsm8k"):
                t = (j.get("results", {}) or {}).get(task)
                if not t:
                    continue
                print(f"  - {task}: {json.dumps(t)}")
            print()


if __name__ == "__main__":
    main()
