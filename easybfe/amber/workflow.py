from __future__ import annotations

import os
import stat
import warnings
from pathlib import Path
from typing import List, Dict
from collections import OrderedDict

from ..config import AmberMdin, AmberStepConfig

RUN_SH_SHEBANG = "#!/usr/bin/env bash"

# Paths that are written into generated commands verbatim, never relativized.
_ALWAYS_ABSOLUTE = {Path(os.devnull)}


def _make_executable(path: Path) -> None:
    """Set user/group/other execute bits on a file (e.g. generated ``run.sh``)."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Step:
    def __init__(
        self,
        config: AmberStepConfig,
        wdir: os.PathLike = '.',
        prmtop: os.PathLike | None = None,
        inpcrd: os.PathLike | None = None,
    ):
        self.config = config
        self.exec = config.exec
        self.name = config.name
        self.wdir = Path(wdir).resolve()
        self.input = ""
        self.set_input()
        self.set_prmtop(prmtop)
        self.set_inpcrd(inpcrd)
    
    def setup_check(self):
        assert self.wdir is not None
        assert self.prmtop is not None
        assert self.inpcrd is not None
        assert self.input != ''
    
    @property
    def outputs(self) -> Dict[str, Path]:
        return {
            'o': self.wdir / f'{self.name}.out',
            'r': self.wdir / f'{self.name}.rst7',
            'inf': self.wdir / f'{self.name}.info',
            'x': self.wdir / f'{self.name}.mdcrd',
            'e': self.wdir / f'{self.name}.mden',
            # 'l': self.wdir / f'{self.name}.log',
            'l': Path(os.devnull)
        }

    def set_prmtop(self, prmtop: os.PathLike):
        self.prmtop = Path(prmtop).resolve() if prmtop else None
    
    def set_inpcrd(self, inpcrd: os.PathLike):
        self.inpcrd = Path(inpcrd).resolve() if inpcrd else None
    
    def set_input(self):
        """Generate the MD input text from the associated step config."""
        mdin = AmberMdin(
            cntrl=self.config.cntrl,
            wt=self.config.wt,
            rst=self.config.rst,
        )
        self.input = mdin.model_dump_mdin()
    
    def write_input(self):
        with open(self.wdir / f"{self.name}.in", 'w') as f:
            f.write(self.input)

    def _resolve_paths(self, use_relpath: bool, relative_to: os.PathLike | None) -> Dict[str, str]:
        """Paths of this step's files, as written into the generated commands.

        Run-tree files are made relative to ``relative_to`` so a leg directory
        can be moved or mounted elsewhere. Device paths such as ``/dev/null``
        (the ``-l`` sink) stay absolute — relativizing those produces a ladder
        of ``../`` whose length depends on how deep the lambda directory sits.
        """
        if use_relpath:
            start = Path(relative_to).resolve() if relative_to is not None else self.wdir

            def render(path: os.PathLike) -> str:
                if Path(path) in _ALWAYS_ABSOLUTE:
                    return str(path)
                return os.path.relpath(path, start)
        else:
            def render(path: os.PathLike) -> str:
                return str(path)

        paths = {
            'i': render(self.wdir / f'{self.name}.in'),
            'p': render(self.prmtop),
            'c': render(self.inpcrd),
            'ref': render(self.inpcrd),
        }
        paths.update({out: render(file) for out, file in self.outputs.items()})
        paths['pdb'] = render(self.wdir / f'{self.name}.pdb')
        return paths

    def create_args(self, use_relpath: bool = True, relative_to: os.PathLike | None = None) -> List[str]:
        """``pmemd`` arguments for this step, without the executable name.

        This is the single source of truth for the command line; both renderers
        below build on it, so the shell script and the groupfile entry can never
        drift apart.
        """
        paths = self._resolve_paths(use_relpath, relative_to)
        args = ['-O']
        for flag in ('i', 'p', 'c', 'ref', *self.outputs):
            args += [f'-{flag}', paths[flag]]
        return args

    def render_groupfile_line(self, use_relpath: bool = True, relative_to: os.PathLike | None = None) -> str:
        """One groupfile entry. Must stay on a single line — that is the format."""
        return ' '.join(self.create_args(use_relpath, relative_to))

    def render_shell(self, use_relpath: bool = True, relative_to: os.PathLike | None = None, export_pdb: bool = True) -> str:
        """Standalone shell snippet running this step, one argument per line.

        Written to ``<name>.sh`` and kept readable on purpose: this file is the
        entry point for debugging a single lambda window by hand.
        """
        paths = self._resolve_paths(use_relpath, relative_to)
        args = self.create_args(use_relpath, relative_to)

        lines = [f'{self.exec} \\']
        # args alternate flag/value after the leading -O
        lines.append(f'  {args[0]} \\')
        for flag, value in zip(args[1::2], args[2::2]):
            lines.append(f'  {flag} {value} \\')
        lines[-1] = lines[-1].rstrip(' \\')

        if export_pdb:
            # `&&` (not a newline): ambpdb must not run on a failed pmemd.
            lines[-1] += ' \\'
            lines.append(f'  && ambpdb -p {paths["p"]} -c {paths["r"]} > {paths["pdb"]}')
        return '\n'.join(lines) + '\n'

    def create(self, use_relpath: bool = True, relative_to: os.PathLike | None = None, export_pdb = True):
        self.setup_check()
        self.wdir.mkdir(exist_ok=True)
        self.write_input()
        with open(self.wdir / f'{self.name}.sh', 'w') as f:
            f.write(self.render_shell(use_relpath, relative_to, export_pdb))

    def link_prev_step(self, step):
        self.prev_step = step
        if self.prmtop is None:
            self.set_prmtop(self.prev_step.prmtop)
        assert self.prev_step.prmtop == self.prmtop, "Not the same topology"
        self.set_inpcrd(self.prev_step.outputs['r'])


class Workflow:
    """The ordered stages of one simulation unit (one lambda window, or a plain MD run)."""

    def __init__(self, wdir: os.PathLike, prmtop: os.PathLike, inpcrd: os.PathLike, steps: List[Step]):
        self.wdir = Path(wdir).resolve()
        self.steps = OrderedDict()
        steps[0].set_inpcrd(inpcrd)
        for i, step in enumerate(steps):
            step.set_prmtop(prmtop)
            step.wdir = self.wdir / step.name
            self.steps[step.name] = step
            if i > 0:
                step.link_prev_step(steps[i - 1])

    def create(self, **kwargs):
        """Write each stage's ``.sh`` plus a ``run.sh`` that runs them in order.

        For an alchemical leg this per-lambda ``run.sh`` is not what executes the
        leg (the leg-level script drives the lambda windows together through
        groupfiles) — it is kept because it is the way to re-run a single lambda
        window by hand when debugging.
        """
        self.wdir.mkdir(exist_ok=True)
        for name, step in self.steps.items():
            step.create(**kwargs)
        run_sh = self.wdir / "run.sh"
        with open(run_sh, "w") as f:
            f.write(RUN_SH_SHEBANG + "\n\n")
            for name, step in self.steps.items():
                f.write(f"cd {name}\n")
                f.write(f"echo Running {name} && touch running.tag\n")
                f.write("if [ ! -f done.tag ]; then\n")
                f.write(f"  source {name}.sh > {name}.stdout 2>&1\n")
                f.write("  if [ $? -ne 0 ]; then\n")
                f.write('    mv running.tag error.tag && echo "Error occurs!" && exit 1\n')
                f.write("  fi\n")
                f.write("  mv running.tag done.tag\n")
                f.write("fi\n")
                f.write("cd ..\n\n")
        _make_executable(run_sh)
        # TODO: generate a real submission script (`#SBATCH` header built from
        # the run configuration) so `sbatch` works without hand-writing one.
        # The previous `run.submit` was dropped: `header` never received a value
        # from any caller, so it only ever held a shebang and `./run.sh`.


def create_groupfile_from_steps(steps: List[Step], dirname: os.PathLike | None = None, fpath: os.PathLike | None = None):
    """Write one groupfile line per step (the AMBER ``-groupfile`` format)."""
    if dirname is not None:
        relative_to = Path(dirname).resolve()
        use_relpath = True
    else:
        use_relpath = False
        relative_to = None

    cmd = '\n'.join(step.render_groupfile_line(use_relpath, relative_to) for step in steps)
    if fpath:
        with open(fpath, 'w') as f:
            f.write(cmd)
    return cmd


# ----------------------------------------------------------------------------
# Body of the generated per-leg ``run.sh``.
#
# Contract: the generated script depends on nothing but bash and AMBER. easybfe
# writes it and reads its status files, but is never needed to execute it — the
# leg has to run inside a plain AMBER image (see tests/test_leg_script.py, which
# enforces this).
#
# The script owns the tag state machine so a leg can resume by itself:
#   <stage>.done.tag   one per completed stage, drives resumption
#   preprod.done.tag   written after the second-to-last stage (early-stop phase)
#   done.tag           written after the final stage
#   running/error/killed.tag   leg-level, as before
# easybfe only reads these; it never writes or clears them (use --force).
# ----------------------------------------------------------------------------

_LEG_USAGE = r'''
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the MD stages of this leg. Safe to re-run: stages that already have a
<stage>.done.tag are skipped.

Options:
  --from STAGE     start at STAGE (default: first)
  --until STAGE    stop after STAGE (default: last)
  --force          clear error/running/killed tags and run anyway
  --list           list the stages of this leg and exit
  -h, --help       show this help

Stages: ${STAGES[*]}
EOF
}
'''

# JSON is written by hand because the leg must not depend on python.
_LEG_STATUS = r'''
json_escape() {
  sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\r//g' -e 's/\t/ /g' \
    | awk 'BEGIN{ORS=""} {print (NR>1 ? "\\n" : "") $0}'
}

# What to show when a stage fails: the first suspicious lines (the root cause is
# usually near the top -- e.g. `cuStreamCreate return value: 709` precedes the
# MPI_Init abort it causes) followed by the tail for context.
error_excerpt() {
  local log="$1" hits text
  [ -f "$log" ] || return 0
  # pmemd writes NUL bytes into its stdout often enough that grep decides the
  # file is binary and reports "Binary file ... matches" instead of the error
  # itself. Strip the NULs and force text mode so the excerpt stays readable
  # (and stays valid JSON once escaped).
  text=$(tr -d '\000' < "$log")
  hits=$(printf '%s\n' "$text" \
    | grep -E -i -a 'error|failed|abort|cannot|unable|invalid|not found|return value' \
    | head -n 6)
  if [ -n "$hits" ]; then printf '%s\n' "$hits"; fi
  printf '%s\n' "$(printf '%s\n' "$text" | tail -n 6)"
}

# Rewritten after every stage transition so the file is accurate even if the
# job is killed mid-stage.
write_status() {
  local state="$1" stage="$2" rc="${3:-0}" log="${4:-}" tmp="$WDIR/.status.json.tmp" first=1 s
  {
    printf '{\n'
    printf '  "leg": "%s",\n' "$LEG"
    printf '  "state": "%s",\n' "$state"
    printf '  "stage": "%s",\n' "$stage"
    printf '  "exit_code": %s,\n' "$rc"
    printf '  "host": "%s",\n' "$(hostname)"
    printf '  "started": "%s",\n' "$STARTED"
    printf '  "updated": "%s",\n' "$(date -Is)"
    printf '  "stages": {'
    for s in "${STAGES[@]}"; do
      [ $first -eq 1 ] || printf ','
      first=0
      if [ -f "$WDIR/$s.done.tag" ]; then printf '"%s": "done"' "$s"
      elif [ "$s" = "$stage" ] && [ "$state" = "failed" ]; then printf '"%s": "failed"' "$s"
      elif [ "$s" = "$stage" ] && [ "$state" = "running" ]; then printf '"%s": "running"' "$s"
      else printf '"%s": "pending"' "$s"; fi
    done
    printf '},\n'
    if [ -n "$log" ]; then
      printf '  "log": "%s",\n' "$(basename "$log")"
      printf '  "error_excerpt": "%s"\n' "$(error_excerpt "$log" | json_escape)"
    else
      printf '  "log": null,\n'
      printf '  "error_excerpt": null\n'
    fi
    printf '}\n'
  } > "$tmp" && mv "$tmp" "$WDIR/status.json"
}
'''

# ----------------------------------------------------------------------------
# CUDA MPS.
#
# A leg runs many ranks over few GPUs (``mpirun -np <ranks>`` with
# ``--gpu-bind=none``); under H-REMD they also synchronize every few hundred
# steps, so without MPS the ranks sharing a GPU time-slice in lockstep and the
# production stage runs several times slower. Each leg manages its own daemon on
# purpose: if a leg dies it can leave the MPS server in an unknown state, and a
# fresh daemon per leg contains that blast radius.
#
# Teardown order matters and is the fix for the "MPS is polluted after a failed
# leg" symptom: leftover pmemd clients keep the daemon alive, so `quit` blocks
# and the pipe directory gets removed while the daemon is still running — the
# next leg then starts a second daemon that fights the first over the per-user
# MPS server. So: stop our clients, then quit with a timeout, then verify the
# daemon is really gone, and only then remove the directories.
#
# Clients are identified two ways, both scoped to processes this script owns:
# our own process descendants, and the client list of our own daemon (which can
# only contain our processes, since the pipe directory is private to this run).
# No pattern-killing by name — that would hit other jobs of the same user.
#
# ``OMPI_MCA_accelerator=null`` is set whenever MPS is active: Open MPI's CUDA
# accelerator component creates a stream inside ``MPI_Init`` which fails against
# an MPS server with ``cuStreamCreate return value: 709``; some ranks then fall
# back to a different pml and ``MPI_Init`` aborts with "at least one MPI process
# is unreachable". pmemd stages MPI buffers on the host, so nothing here needs
# CUDA-aware MPI. Escape hatches: EASYBFE_DISABLE_MPS=1 skips MPS entirely,
# EASYBFE_KEEP_OMPI_ACCELERATOR=1 leaves the component alone.
# ----------------------------------------------------------------------------
_LEG_MPS = r'''
EASYBFE_MPS_STARTED=0
EASYBFE_MPS_DAEMON_PID=""

disable_ompi_cuda_accelerator() {
  if [ "${EASYBFE_KEEP_OMPI_ACCELERATOR:-0}" = "1" ]; then return 0; fi
  export OMPI_MCA_accelerator=null
}

mps_control() { timeout 15 nvidia-cuda-mps-control "$@"; }

# PIDs descended from this script, deepest first.
descendant_pids() {
  local parent="$1" child
  for child in $(pgrep -P "$parent" 2>/dev/null); do
    descendant_pids "$child"
    echo "$child"
  done
}

start_mps() {
  if [ "${EASYBFE_DISABLE_MPS:-0}" = "1" ]; then
    echo "CUDA MPS: disabled (EASYBFE_DISABLE_MPS=1)"
    return 0
  fi
  if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ -d "${CUDA_MPS_PIPE_DIRECTORY}" ]; then
    disable_ompi_cuda_accelerator
    echo "CUDA MPS: reusing external daemon at ${CUDA_MPS_PIPE_DIRECTORY}"
    return 0
  fi
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "CUDA MPS: unavailable (nvidia-cuda-mps-control not found)"
    return 0
  fi
  # Only one MPS server exists per user per GPU. If a daemon of ours is already
  # running but was not handed to us, starting a second one would fight it.
  # Leaving it alone is the conservative choice; say so loudly and run without.
  local existing
  existing=$(pgrep -u "$USER" -f nvidia-cuda-mps-control 2>/dev/null | tr '\n' ' ')
  if [ -n "$existing" ]; then
    echo "CUDA MPS: a daemon for $USER is already running (pid: $existing) but"
    echo "          CUDA_MPS_PIPE_DIRECTORY is not exported. Not starting a second"
    echo "          daemon; running without MPS (slower). Export that variable to"
    echo "          reuse it, or stop the stale daemon."
    return 0
  fi

  export CUDA_MPS_PIPE_DIRECTORY=$(mktemp -d "/tmp/nvidia-mps-pipe-${USER}-$$-XXXXXX")
  export CUDA_MPS_LOG_DIRECTORY=$(mktemp -d "/tmp/nvidia-mps-log-${USER}-$$-XXXXXX")
  nvidia-cuda-mps-control -d
  sleep 5
  EASYBFE_MPS_DAEMON_PID=$(pgrep -u "$USER" -f nvidia-cuda-mps-control 2>/dev/null | head -1)

  # A daemon that answers is worth using; one that does not would fail every
  # rank at MPI_Init, so degrade to no MPS rather than lose the leg.
  if ! echo get_server_list | mps_control >/dev/null 2>&1; then
    echo "CUDA MPS: daemon did not come up healthy; continuing without MPS"
    stop_mps
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    return 0
  fi
  EASYBFE_MPS_STARTED=1
  disable_ompi_cuda_accelerator
  echo "CUDA MPS: started daemon (pid ${EASYBFE_MPS_DAEMON_PID:-?}) at ${CUDA_MPS_PIPE_DIRECTORY}"
}

stop_mps() {
  if [ "${EASYBFE_MPS_STARTED}" != "1" ] && [ -z "${EASYBFE_MPS_DAEMON_PID}" ]; then
    return 0
  fi

  # 1. Our own leftover clients, or `quit` below will block on them.
  local pid srv cli
  for pid in $(descendant_pids $$); do kill -TERM "$pid" 2>/dev/null; done
  for srv in $(echo get_server_list | mps_control 2>/dev/null); do
    for cli in $(echo get_client_list "$srv" | mps_control 2>/dev/null); do
      kill -TERM "$cli" 2>/dev/null
    done
  done
  sleep 2

  # 2. Ask the daemon to quit, then 3. make sure it actually did. `quit` returns
  #    as soon as the request is accepted -- the daemon needs another second or
  #    two to go away -- so poll instead of checking once, or we SIGKILL a daemon
  #    that is shutting down cleanly and warn about it on every single leg.
  echo quit | mps_control >/dev/null 2>&1
  if [ -n "${EASYBFE_MPS_DAEMON_PID}" ]; then
    local waited=0
    while [ "${waited}" -lt 10 ] && kill -0 "${EASYBFE_MPS_DAEMON_PID}" 2>/dev/null; do
      sleep 1
      waited=$((waited + 1))
    done
    # A daemon still alive here is wedged: leaving it would make the next leg
    # start a second daemon for the same user+GPU, and every rank of that leg
    # would then die in cudaSetDevice with "out of memory".
    if kill -0 "${EASYBFE_MPS_DAEMON_PID}" 2>/dev/null; then
      kill -KILL "${EASYBFE_MPS_DAEMON_PID}" 2>/dev/null
      echo "CUDA MPS: daemon ${EASYBFE_MPS_DAEMON_PID} did not quit; killed it"
    fi
  fi

  # 4. Only now is it safe to drop the pipe directory.
  rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null
  EASYBFE_MPS_STARTED=0
  EASYBFE_MPS_DAEMON_PID=""
  echo "CUDA MPS: stopped"
}
'''

_LEG_RUNNERS = r'''
# Non-MPI stage: run it once per lambda window, sequentially.
run_seq_stage() {
  local name="$1" rc=0 d
  for d in "$WDIR"/lambda*/"$name"; do
    [ -d "$d" ] || continue
    if ! ( cd "$d" && source "./$name.sh" > "$name.stdout" 2>&1 ); then
      rc=$?
      echo "  failed in ${d#$WDIR/} (see $name.stdout)"
      return $rc
    fi
  done
  return 0
}

# MPI stage: all lambda windows in one mpirun via the groupfile.
run_mpi_stage() {
  local idx="$1" name="$2" log="$WDIR/$name.stdout"
  local cmd=(mpirun -np "$NPROCS" "${STAGE_EXEC[$idx]}" -ng "$NGROUPS" -groupfile "$name.groupfile")
  if [ "${STAGE_REMD[$idx]}" = "1" ]; then cmd+=(-rem 3 -remlog "$name.log"); fi
  echo "  ${cmd[*]}"
  ( cd "$WDIR" && "${cmd[@]}" ) > "$log" 2>&1
}
'''


def _build_leg_script(
    workflows: List[Workflow],
    wdir: Path,
    nprocs: int,
    step_names: tuple,
) -> str:
    """Build the single ``run.sh`` that runs one leg.

    Parameters
    ----------
    workflows : list of Workflow
        Per-lambda workflows, all sharing the same stage names.
    wdir : pathlib.Path
        Leg directory the script is written to and executed from.
    nprocs : int
        Total MPI ranks for the grouped ``pmemd.cuda.MPI`` stages.
    step_names : tuple of str
        Stage names, in execution order.
    """
    steps = workflows[0].steps
    execs = []
    for name in step_names:
        cfg = steps[name].config
        if cfg.use_mpi:
            execs.append(cfg.exec if cfg.exec.endswith('.MPI') else f'{cfg.exec}.MPI')
        else:
            execs.append(cfg.exec)

    def sh_array(name: str, values) -> str:
        return f'{name}=({" ".join(str(v) for v in values)})'

    header = [
        RUN_SH_SHEBANG,
        '# Generated by easybfe. Depends on bash + AMBER only -- do not add a',
        '# dependency on easybfe here (see the leg-script contract in workflow.py).',
        '',
        'WDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"',
        'LEG="$(basename "$WDIR")"',
        'STARTED="$(date -Is)"',
        f'NPROCS={nprocs}',
        f'NGROUPS={len(workflows)}',
        sh_array('STAGES', step_names),
        sh_array('STAGE_EXEC', execs),
        sh_array('STAGE_MPI', [1 if steps[n].config.use_mpi else 0 for n in step_names]),
        sh_array('STAGE_REMD', [1 if steps[n].config.use_remd else 0 for n in step_names]),
        # Tags kept for the early-stop orchestration and for `easybfe abfe analyze`.
        f'PREPROD_STAGE="{step_names[-2] if len(step_names) > 1 else ""}"',
        f'FINAL_STAGE="{step_names[-1]}"',
    ]

    main = r'''
FROM=""; UNTIL=""; FORCE=0; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from)  FROM="$2";  shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --force) FORCE=1;    shift ;;
    --list)  LIST=1;     shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stage_index() {
  local want="$1" i
  for i in "${!STAGES[@]}"; do
    if [ "${STAGES[$i]}" = "$want" ]; then echo "$i"; return 0; fi
  done
  echo "Unknown stage: $want (have: ${STAGES[*]})" >&2
  return 1
}

if [ $LIST -eq 1 ]; then
  for i in "${!STAGES[@]}"; do
    tag="pending"; [ -f "$WDIR/${STAGES[$i]}.done.tag" ] && tag="done"
    printf '%s\t%s\tmpi=%s\tremd=%s\n' "${STAGES[$i]}" "$tag" "${STAGE_MPI[$i]}" "${STAGE_REMD[$i]}"
  done
  exit 0
fi

START_IDX=0; END_IDX=$(( ${#STAGES[@]} - 1 ))
if [ -n "$FROM" ];  then START_IDX=$(stage_index "$FROM")  || exit 2; fi
if [ -n "$UNTIL" ]; then END_IDX=$(stage_index "$UNTIL")   || exit 2; fi
if [ "$START_IDX" -gt "$END_IDX" ]; then
  echo "--from ${STAGES[$START_IDX]} is after --until ${STAGES[$END_IDX]}" >&2
  exit 2
fi

cd "$WDIR" || exit 1

if [ $FORCE -eq 1 ]; then
  rm -f "$WDIR/error.tag" "$WDIR/running.tag" "$WDIR/killed.tag"
fi
if [ -f "$WDIR/running.tag" ]; then
  echo "Found running.tag: a run may still be in progress. Use --force to override."
  exit 0
fi
if [ -f "$WDIR/error.tag" ]; then
  echo "Found error.tag: the previous run failed. Fix it and re-run with --force."
  exit 0
fi

CLEANED=0
cleanup() {
  local rc=$?
  [ $CLEANED -eq 1 ] && return
  CLEANED=1
  stop_mps
  if [ -f "$WDIR/running.tag" ]; then
    if [ $rc -eq 0 ]; then rm -f "$WDIR/running.tag"; else mv "$WDIR/running.tag" "$WDIR/killed.tag"; fi
  fi
}
trap cleanup EXIT TERM INT HUP

touch "$WDIR/running.tag"
write_status running "${STAGES[$START_IDX]}"
start_mps
start_seconds=$(date +%s)

for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
  name="${STAGES[$idx]}"
  if [ -f "$WDIR/$name.done.tag" ]; then
    echo "Skipping $name (already done)"
    continue
  fi

  echo "Running $name"
  write_status running "$name"
  if [ "${STAGE_MPI[$idx]}" = "1" ]; then
    run_mpi_stage "$idx" "$name"
  else
    run_seq_stage "$name"
  fi
  rc=$?

  if [ $rc -ne 0 ]; then
    log="$WDIR/$name.stdout"
    echo "Error: stage $name failed with exit code $rc"
    # Surface the reason here as well: whoever launched this script sees only
    # its stdout, and the interesting lines are otherwise buried in the log.
    if [ -f "$log" ]; then
      echo "----- $name.stdout (excerpt) -----"
      error_excerpt "$log"
      echo "----------------------------------"
    fi
    write_status failed "$name" "$rc" "$log"
    rm -f "$WDIR/running.tag"
    touch "$WDIR/error.tag"
    exit $rc
  fi

  touch "$WDIR/$name.done.tag"
  if [ -n "$PREPROD_STAGE" ] && [ "$name" = "$PREPROD_STAGE" ]; then
    touch "$WDIR/preprod.done.tag"
  fi
  if [ "$name" = "$FINAL_STAGE" ]; then
    touch "$WDIR/done.tag"
  fi
done

write_status completed "${STAGES[$END_IDX]}"
stop_mps
rm -f "$WDIR/running.tag"

duration=$(( $(date +%s) - start_seconds ))
printf 'Execution time: %d h %d min %d sec\n' \
  $(( duration / 3600 )) $(( (duration % 3600) / 60 )) $(( duration % 60 ))
'''

    return '\n'.join(header) + '\n' + _LEG_USAGE + _LEG_STATUS + _LEG_MPS + _LEG_RUNNERS + main


def create_script_for_workflows(workflows: List[Workflow], wdir: os.PathLike, nprocs: int = -1):
    """Generate the leg's ``run.sh`` and the per-stage groupfiles.

    One script per leg handles every phase; the caller selects stages with
    ``run.sh [--from STAGE] [--until STAGE]``. The early-stop orchestration in
    :mod:`easybfe.abfe.piepline` uses ``--until <second-to-last>`` followed by
    ``--from <last>``; a plain ``run.sh`` runs the whole leg.

    The script owns the tag state machine, so re-running it resumes at the first
    stage without a ``<stage>.done.tag``. It requires only bash and AMBER.
    """
    for wf in workflows:
        wf.create()
    step_names = tuple([name for name in workflows[0].steps])
    for wf in workflows[1:]:
        this_wf_steps = tuple([name for name in wf.steps])
        if this_wf_steps != step_names:
            warnings.warn(f"Not same workflow {step_names} != {this_wf_steps}")

    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)

    if nprocs < 0:
        nprocs = len(workflows)
    else:
        nprocs = max(1, nprocs // len(workflows)) * len(workflows)

    for name in step_names:
        if workflows[0].steps[name].config.use_mpi:
            create_groupfile_from_steps([wf.steps[name] for wf in workflows], wdir, wdir / f'{name}.groupfile')

    run_sh = wdir / "run.sh"
    run_sh.write_text(_build_leg_script(workflows, wdir, nprocs, step_names))
    _make_executable(run_sh)
