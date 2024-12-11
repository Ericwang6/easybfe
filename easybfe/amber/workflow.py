import os
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict
from .settings import AmberMdin
from ..cmd import run_command, set_directory


class Step:
    def __init__(
        self, 
        name: str, 
        wdir: os.PathLike = '.', 
        mdin: os.PathLike | str | AmberMdin = '',
        prmtop: os.PathLike | None = None, 
        inpcrd: os.PathLike | None = None,
        exec: str = 'pmemd.cuda'
    ):
        self.exec = exec
        self.name = name
        self.wdir = Path(wdir).resolve()
        self.input = ""
        self.set_input(mdin)
        if prmtop is not None:
            self.set_prmtop(prmtop)
        if inpcrd is not None:
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
        self.prmtop = Path(prmtop).resolve()
    
    def set_inpcrd(self, inpcrd: os.PathLike):
        self.inpcrd = Path(inpcrd).resolve()
    
    def set_input(self, mdin: os.PathLike | str | AmberMdin):
        if isinstance(mdin, AmberMdin):
            self.input = mdin.model_dump_mdin()
        elif os.path.isfile(mdin):
            with open(mdin) as f:
                self.input = f.read()
        else:
            self.input = mdin
    
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
        with open(self.wdir / 'run.sh', 'w') as f:
            for name, step in self.steps.items():
                f.write(f'cd {name}\n')
                f.write(f'echo Running {name} && touch running.tag\n')
                f.write('if [ ! -f done.tag ]; then\n')
                f.write(f'  source {name}.sh > {name}.stdout 2>&1\n')
                f.write('  if [ $? -ne 0 ]; then\n')
                f.write('    mv running.tag error.tag && echo "Error occurs!" && exit 1\n')
                f.write('  fi\n')
                f.write('  mv running.tag done.tag')
                f.write('fi\n')
                f.write('cd ..\n\n')
        with open(self.wdir / 'run.submit', 'w') as f:
            f.write(self.header + '\n')
            f.write('source run.sh')
        
    def submit(self, platform: str = "slurm"):
        if platform == 'slurm':
            with set_directory(self.wdir):
                run_command(['sbatch', 'run.submit'])
        elif platform == 'local':
            with set_directory(self.wdir):
                run_command(['source run.submit'])
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
