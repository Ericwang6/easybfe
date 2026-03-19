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
            'l': self.wdir / f'{self.name}.log',
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


def create_script_for_workflows(workflows: List[Workflow], wdir: os.PathLike, nprocs: int = -1):
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
    
    afunc = '''
run_step_seq() {
  local step_dir="$1"
  local name="$2"

  cd "$step_dir" || return 1

  echo "Running $step_dir ..."
  source "$name.sh" > "$name.stdout" 2>&1
  local rc=$?

  if [ $rc -ne 0 ]; then
    mv "$WDIR/running.tag" "$WDIR/error.tag"
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
    if [ -f $WDIR/running.tag ]; then mv $WDIR/running.tag $WDIR/killed.tag; fi
    echo "[`date`] Cleanup done."
    exit 2 
}
trap cleanup TERM INT HUP
'''

    script_lines = [
        RUN_SH_SHEBANG,
        "",
        'WDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"',
        afunc,
        cleanup_func,
        r'start=$(date +%s)',
        (
            'if [ -f done.tag ] || [ -f running.tag ] || [ -f error.tag ]; then\n'
            '  exit 0\n'
            'fi\n'
            'touch running.tag'
        ),
    ]
    for name in step_names:
        cfg = workflows[0].steps[name].config
        pmemd_exec = cfg.exec if cfg.exec.endswith('.MPI') else f'{cfg.exec}.MPI'
        script_lines.append(f'\necho "Running {name}"')
        if cfg.use_mpi:
            create_groupfile_from_steps([wf.steps[name] for wf in workflows], wdir, wdir / f'{name}.groupfile')
            cmd = f"mpirun -np {nprocs} {pmemd_exec} -ng {len(workflows)} -groupfile {name}.groupfile"
            cmd += f' -rem 3 -remlog {name}.log' if cfg.use_remd else ''
            script_lines.append(cmd)
            script_lines.append((
                'if [ $? -ne 0 ]; then\n'
                '  mv running.tag error.tag && echo "Error occurs!"\n'
                '  exit 1\n'
                'fi\n'
            ))
        else:
            for n in range(len(workflows)):
                script_lines.append(f'run_step_seq {os.path.relpath(workflows[n].steps[name].wdir, wdir)} {name} || exit 1')
    
    script_lines += [
        '\n',
        'mv running.tag done.tag\n',
        r"end=$(date +%s)",
        "duration=$((end - start))\n",
        'hours=$(( duration / 3600 ))',
        'minutes=$(( (duration % 3600) / 60 ))',
        'seconds=$(( duration % 60 ))\n',
        r'echo "Execution time: ${hours} h ${minutes} min ${seconds} sec"'
    ]
    run_sh = wdir / "run.sh"
    with open(run_sh, "w") as f:
        f.write("\n".join(script_lines))
    _make_executable(run_sh)

