#!/usr/bin/env python3
"""Report pmemd's own ns/day for a finished ABFE run, read from mdout.

    prod-rate.py <run-dir> [--stage 05.prod] [--csv out.csv]

`<run-dir>` is the directory `easybfe abfe pipeline -o` wrote, i.e. the one
containing `abfe/{solvent,complex,restraint}/lambda*/`. Each lambda window's
stage writes `<stage>.out` (AMBER's mdout), whose final block is

    | Average timings for all steps:
    |     Elapsed(s) =     129.28 Per Step(ms) =       2.07
    |         ns/day =     167.14   seconds/ns =     516.93

Two things this is careful about:

* mdout contains **two** ns/day figures. The first is "Average timings for
  last N steps" -- an instantaneous rate over as little as one step, which can
  be off by 10x. Only the "all steps" block is a rate; that is what is read.
* The number is *per lambda window*. All 24 windows run concurrently, packed
  several to a GPU, so the node's aggregate throughput is the sum, not the mean.
  Both are printed.

Reads `.out` (mdout), not `.info` (mdinfo) -- this used to read `.info`, and
that was wrong specifically for fast/small systems (the solvent leg here).
Under REMD (`icfe=1`), AMBER rewrites `.info` at every exchange
(`bar_intervall` == `nstlim`, e.g. every 125 steps), and each rewrite appears
to pay a small fixed synchronization/flush cost. For a slow-per-step leg
(complex/restraint here, ~2.7 ms/step) that 125-step interval is long in wall
clock, so the fixed cost is negligible. For a fast-per-step leg (solvent,
~1 ms/step) the same interval is short, so the same fixed cost is a much
larger fraction of it -- inflating `.info`'s reported per-step time and
understating its ns/day by ~15-20%, reproduced on multiple lambda windows.
`.out`'s "all steps" block is printed once, at the true end of the whole
stage, over the *entire* run rather than one exchange interval, and does not
show this effect. See
`container/e2e-abfe-test/results/a100x4-nccl-mpi5-localssd/README.md` §4 for
the `.info`-vs-`.out` numbers this was diagnosed from. complex/restraint are
unaffected either way (`.info` and `.out` agreed there to 2 decimal places);
this only matters for legs/systems fast enough that a 125-step interval is a
short fraction of a second.
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

ALL_STEPS = re.compile(r"Average timings for all steps", re.I)
NS_PER_DAY = re.compile(r"ns/day\s*=\s*([0-9.]+)")


def rate_from_mdout(path: Path) -> float | None:
    """ns/day from the 'all steps' block, or None if the run did not finish."""
    seen_header = False
    for line in path.read_text(errors="replace").splitlines():
        if ALL_STEPS.search(line):
            seen_header = True
        elif seen_header:
            m = NS_PER_DAY.search(line)
            if m:
                return float(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--stage", default="05.prod",
                    help="stage name to report (default: 05.prod)")
    ap.add_argument("--label", default="", help="tag rows with a machine name")
    ap.add_argument("--csv", type=Path, help="also write per-window rows here")
    args = ap.parse_args()

    outs = sorted(args.run_dir.glob(f"**/{args.stage}.out"))
    if not outs:
        print(f"no {args.stage}.out under {args.run_dir}", file=sys.stderr)
        return 1

    rows = []
    for out in outs:
        # EasyBFE writes .../abfe/<leg>/<window>/<stage>/<stage>.out -- the
        # stage is a directory, not just a filename prefix. Walk up from the
        # file rather than assuming a depth, and recognise the leg by name so a
        # different nesting cannot silently relabel windows as legs.
        parts = out.parts
        leg = next((p for p in reversed(parts)
                    if p in ("solvent", "complex", "restraint")), "")
        if leg:
            window = parts[parts.index(leg) + 1]
        else:                       # unknown layout: best effort
            leg, window = parts[-4], parts[-3]
        rate = rate_from_mdout(out)
        rows.append({"label": args.label, "leg": leg, "window": window,
                     "stage": args.stage,
                     "ns_per_day": "" if rate is None else f"{rate:.2f}"})

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    hdr = (f"{'leg':<12} {'n':>3} {'mean':>9} {'min':>9} {'max':>9} "
           f"{'node total':>12}")
    print(f"\n{args.stage}  --  pmemd ns/day per lambda window"
          + (f"  [{args.label}]" if args.label else ""))
    print(hdr)
    print("-" * len(hdr))
    legs = sorted({r["leg"] for r in rows})
    for leg in legs:
        vals = [float(r["ns_per_day"]) for r in rows
                if r["leg"] == leg and r["ns_per_day"]]
        missing = sum(1 for r in rows if r["leg"] == leg and not r["ns_per_day"])
        if not vals:
            print(f"{leg:<12} {'-':>3}  (no completed windows)")
            continue
        print(f"{leg:<12} {len(vals):>3} {statistics.fmean(vals):>9.1f} "
              f"{min(vals):>9.1f} {max(vals):>9.1f} {sum(vals) / 1000:>10.2f} us/day"
              + (f"   ({missing} window(s) without a timing block)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
