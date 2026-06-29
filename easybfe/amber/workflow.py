from __future__ import annotations

import os
import stat
import warnings
from pathlib import Path
from typing import List, Dict
from collections import OrderedDict

from ..config import AmberMdin, AmberStepConfig
from ..cmd import run_command, set_directory

RUN_SH_SHEBANG = "#!/usr/bin/env bash"


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

    def create_cmd(self, use_relpath: bool = True, relative_to: os.PathLike | None = None, export_pdb = True):
        if use_relpath:
            start = Path(relative_to).resolve() if relative_to is not None else self.wdir
            prmtop = os.path.relpath(self.prmtop, start)
            inpcrd = os.path.relpath(self.inpcrd, start)
            outputs = {out: os.path.relpath(file, start) for out, file in self.outputs.items()}
            input = os.path.relpath(self.wdir / f'{self.name}.in', start)
            pdb = os.path.relpath(self.wdir / f'{self.name}.pdb', start)
        else:
            prmtop = self.prmtop
            inpcrd = self.inpcrd
            outputs = self.outputs
            input = self.wdir / f'{self.name}.in'
            pdb = self.wdir / f'{self.name}.pdb'
        
        cmd = f'{self.exec} -O -i {input} -p {prmtop} -c {inpcrd} -ref {inpcrd} '
        for out, file in outputs.items():
            cmd += f'-{out} {file} '
        if export_pdb:
            cmd += f' && ambpdb -p {prmtop} -c {outputs["r"]} > {pdb}'
        return cmd

    def create(self, use_relpath: bool = True, relative_to: os.PathLike | None = None, export_pdb = True):
        self.setup_check()
        self.wdir.mkdir(exist_ok=True)
        self.write_input()
        cmd = self.create_cmd(use_relpath, relative_to, export_pdb)
        with open(self.wdir / f'{self.name}.sh', 'w') as f:
            f.write('\n'.join(cmd.split(' && ')))

    def link_prev_step(self, step):
        self.prev_step = step
        if self.prmtop is None:
            self.set_prmtop(self.prev_step.prmtop)
        assert self.prev_step.prmtop == self.prmtop, "Not the same topology"
        self.set_inpcrd(self.prev_step.outputs['r'])


class Workflow:
    def __init__(self, wdir: os.PathLike, prmtop: os.PathLike, inpcrd: os.PathLike, steps: List[Step], header: List[str] | str = ""):
        self.wdir = Path(wdir).resolve()
        self.steps = OrderedDict()
        steps[0].set_inpcrd(inpcrd)
        for i, step in enumerate(steps):
            step.set_prmtop(prmtop)
            step.wdir = self.wdir / step.name
            self.steps[step.name] = step
            if i > 0:
                step.link_prev_step(steps[i - 1])
        
        if isinstance(header, list):
            self.header = '\n'.join(header)
        else:
            self.header = header
    
    def create(self, **kwargs):
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
        with open(self.wdir / "run.submit", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(self.header + "\n")
            f.write("./run.sh\n")
        
    def submit(self, platform: str = "slurm"):
        if platform == 'slurm':
            with set_directory(self.wdir):
                run_command(['sbatch', 'run.submit'])
        elif platform == 'local':
            with set_directory(self.wdir):
                run_command(['source', 'run.submit'])
        else:
            raise NotImplementedError('Unsupported platform')


def create_groupfile_from_steps(steps: List[Step], dirname: os.PathLike | None = None, fpath: os.PathLike | None = None):
    cmds = []
    if dirname is not None:
        relative_to = Path(dirname).resolve() 
        use_relpath = True
    else:
        use_relpath = False
        relative_to = None

    for step in steps:
        cmd = step.create_cmd(use_relpath, relative_to, False)
        cmds.append(' '.join(cmd.split()[1:]))
    
    cmd = '\n'.join(cmds)
    if fpath:
        with open(fpath, 'w') as f:
            f.write(cmd)
    return cmd


# ----------------------------------------------------------------------------
# CUDA MPS helpers injected into every generated leg script.
#
# Multiple alchemical ranks share a small number of GPUs (``--gpu-bind=none`` +
# ``mpirun -np <ranks>``). Routing them through an NVIDIA MPS daemon lets the
# ranks overlap on each GPU, which speeds up the multi-rank ``pmemd.cuda.MPI``
# stages. The block is self-contained: it reuses an externally started daemon
# when ``CUDA_MPS_PIPE_DIRECTORY`` is already exported, starts its own daemon
# when ``nvidia-cuda-mps-control`` is available, and is a no-op (with a clear
# log message) otherwise. Set ``EASYBFE_DISABLE_MPS=1`` to skip MPS entirely
# (used for A/B speed comparisons).
# ----------------------------------------------------------------------------
_MPS_FUNCS = r'''
EASYBFE_MPS_STARTED=0
maybe_start_mps() {
  if [ "${EASYBFE_DISABLE_MPS:-0}" = "1" ]; then
    echo "CUDA MPS: disabled (EASYBFE_DISABLE_MPS=1)"
    return 0
  fi
  if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ -d "${CUDA_MPS_PIPE_DIRECTORY}" ]; then
    echo "CUDA MPS: reused (external daemon at ${CUDA_MPS_PIPE_DIRECTORY})"
    return 0
  fi
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    export CUDA_MPS_PIPE_DIRECTORY=$(mktemp -d "/tmp/nvidia-mps-pipe-${USER}-$$-XXXXXX")
    export CUDA_MPS_LOG_DIRECTORY=$(mktemp -d "/tmp/nvidia-mps-log-${USER}-$$-XXXXXX")
    nvidia-cuda-mps-control -d && sleep 5
    EASYBFE_MPS_STARTED=1
    echo "CUDA MPS: enabled (started daemon at ${CUDA_MPS_PIPE_DIRECTORY})"
  else
    echo "CUDA MPS: unavailable (nvidia-cuda-mps-control not found)"
  fi
  return 0
}
maybe_stop_mps() {
  if [ "${EASYBFE_MPS_STARTED}" = "1" ]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}" 2>/dev/null || true
    EASYBFE_MPS_STARTED=0
    echo "CUDA MPS: stopped"
  fi
}
'''


def _build_leg_script(
    workflows: List[Workflow],
    wdir: Path,
    nprocs: int,
    step_names: tuple,
    tag_prefix: str = "",
) -> str:
    """Build a leg run script body for a subset of workflow stages.

    Parameters
    ----------
    workflows : list of Workflow
        Per-lambda workflows (all sharing the same stage names).
    wdir : pathlib.Path
        Leg directory the script is written to / executed from.
    nprocs : int
        Total MPI ranks for the grouped ``pmemd.cuda.MPI`` stages.
    step_names : tuple of str
        Stage names to run in this script (a subset of the full workflow).
    tag_prefix : str, optional
        Prefix for the run/done/error/killed tag files (e.g. ``"preprod."``).
        Empty string uses the default ``running.tag`` / ``done.tag`` names.
    """
    afunc = '''
run_step_seq() {
  local step_dir="$1"
  local name="$2"

  cd "$step_dir" || return 1

  echo "Running $step_dir ..."
  source "$name.sh" > "$name.stdout" 2>&1
  local rc=$?

  if [ $rc -ne 0 ]; then
    mv "$WDIR/$RUNNING_TAG" "$WDIR/$ERROR_TAG"
    echo "Error occurs in $name (exit code $rc)"
    cd "$WDIR"
    return $rc
  fi

  cd "$WDIR"
}
'''
    cleanup_func = '''
cleanup() {
    echo "[`date`] Caught termination signal. Cleaning up..."
    maybe_stop_mps
    if [ -f $WDIR/$RUNNING_TAG ]; then mv $WDIR/$RUNNING_TAG $WDIR/$KILLED_TAG; fi
    echo "[`date`] Cleanup done."
    exit 2
}
trap cleanup TERM INT HUP
'''

    tag_vars = (
        f'RUNNING_TAG="{tag_prefix}running.tag"\n'
        f'DONE_TAG="{tag_prefix}done.tag"\n'
        f'ERROR_TAG="{tag_prefix}error.tag"\n'
        f'KILLED_TAG="{tag_prefix}killed.tag"'
    )

    script_lines = [
        RUN_SH_SHEBANG,
        "",
        'WDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"',
        tag_vars,
        _MPS_FUNCS,
        afunc,
        cleanup_func,
        r'start=$(date +%s)',
        (
            'if [ -f $RUNNING_TAG ]; then\n'
            '  echo "Found $RUNNING_TAG: a run may still be in progress. Skip this run."\n'
            '  echo "Delete $RUNNING_TAG to force rerun."\n'
            '  exit 0\n'
            'fi\n'
            'if [ -f $DONE_TAG ]; then\n'
            '  echo "Found $DONE_TAG: previous run has finished. Skip this run."\n'
            '  echo "Delete $DONE_TAG to force rerun."\n'
            '  exit 0\n'
            'fi\n'
            'if [ -f $ERROR_TAG ]; then\n'
            '  echo "Found $ERROR_TAG: previous run ended with error. Skip this run."\n'
            '  echo "Delete $ERROR_TAG to force rerun."\n'
            '  exit 0\n'
            'fi\n'
            'touch $RUNNING_TAG\n'
            'maybe_start_mps'
        ),
    ]
    for name in step_names:
        cfg = workflows[0].steps[name].config
        pmemd_exec = cfg.exec if cfg.exec.endswith('.MPI') else f'{cfg.exec}.MPI'
        script_lines.append(f'\necho "Running {name}"')
        if cfg.use_mpi:
            cmd = f"mpirun -np {nprocs} {pmemd_exec} -ng {len(workflows)} -groupfile {name}.groupfile"
            cmd += f' -rem 3 -remlog {name}.log' if cfg.use_remd else ''
            script_lines.append(cmd)
            script_lines.append((
                'if [ $? -ne 0 ]; then\n'
                '  mv $RUNNING_TAG $ERROR_TAG && echo "Error occurs!"\n'
                '  maybe_stop_mps\n'
                '  exit 1\n'
                'fi\n'
            ))
        else:
            for n in range(len(workflows)):
                script_lines.append(f'run_step_seq {os.path.relpath(workflows[n].steps[name].wdir, wdir)} {name} || {{ maybe_stop_mps; exit 1; }}')

    script_lines += [
        '\n',
        'maybe_stop_mps\n',
        'mv $RUNNING_TAG $DONE_TAG\n',
        r"end=$(date +%s)",
        "duration=$((end - start))\n",
        'hours=$(( duration / 3600 ))',
        'minutes=$(( (duration % 3600) / 60 ))',
        'seconds=$(( duration % 60 ))\n',
        r'echo "Execution time: ${hours} h ${minutes} min ${seconds} sec"'
    ]
    return "\n".join(script_lines)


def create_script_for_workflows(workflows: List[Workflow], wdir: os.PathLike, nprocs: int = -1):
    """Generate the leg run scripts for a set of per-lambda workflows.

    Three scripts are written into ``wdir``:

    - ``run.sh`` runs every workflow stage and writes ``done.tag`` (the default
      single-phase behavior).
    - ``run.preprod.sh`` runs all stages except the last (the "pre-production"
      stages) and writes ``preprod.done.tag``. Only emitted when the workflow
      has more than one stage.
    - ``run.prod.sh`` runs only the final stage and writes ``done.tag``; it is
      meant to follow ``run.preprod.sh`` (the production stage reads the
      restart of the previous stage).

    Each script detects and uses CUDA MPS when available (see :data:`_MPS_FUNCS`).
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

    # Pre-create the per-stage groupfiles once (shared by every generated script).
    for name in step_names:
        if workflows[0].steps[name].config.use_mpi:
            create_groupfile_from_steps([wf.steps[name] for wf in workflows], wdir, wdir / f'{name}.groupfile')

    # Full single-phase script.
    run_sh = wdir / "run.sh"
    run_sh.write_text(_build_leg_script(workflows, wdir, nprocs, step_names, tag_prefix=""))
    _make_executable(run_sh)

    # Two-phase scripts for early-stop orchestration (pre-production then production).
    if len(step_names) > 1:
        preprod_sh = wdir / "run.preprod.sh"
        preprod_sh.write_text(
            _build_leg_script(workflows, wdir, nprocs, step_names[:-1], tag_prefix="preprod.")
        )
        _make_executable(preprod_sh)

        prod_sh = wdir / "run.prod.sh"
        prod_sh.write_text(
            _build_leg_script(workflows, wdir, nprocs, step_names[-1:], tag_prefix="")
        )
        _make_executable(prod_sh)

