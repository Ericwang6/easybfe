import os
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict


def write_cntrl(config: Dict[str, Any]):
    num_space = max([len(key) for key in config.keys()]) + 1
    template = '{:<' + str(num_space) + '} = {},' 
    lines = ['&cntrl']
    for key in config:
        if isinstance(config[key], str):
            assert "'" not in config[key]
            value = f"'{config[key]}'"
        else:
            value = config[key]
        lines.append(template.format(key, value))
    lines.append('/')
    return '\n'.join(lines)


class Step:
    def __init__(self, name: str, dirname: os.PathLike | None, exec: str = 'pmemd.cuda', prmtop: os.PathLike | None = None, inpcrd: os.PathLike | None = None):
        self.exec = exec
        self.name = name
        if dirname is not None:
            self.set_parent_dir(dirname)
        if prmtop is not None:
            self.set_prmtop(prmtop)
        if inpcrd is not None:
            self.set_inpcrd(inpcrd)
    
    def set_parent_dir(self, dirname: os.PathLike):
        self.dirname = Path(dirname).resolve()
        self.wdir = self.dirname / self.name
        self.outputs = {
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
    
    def set_input(self, input: os.PathLike | str):
        if os.path.isfile(input):
            with open(input) as f:
                self.input = f.read()
        else:
            self.input = input
    
    def write_input(self):
        with open(self.dirname / f"{self.name}.in", 'w') as f:
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
            cmd += f'-{out} {os.path.realpath(file, self.wdir)}'
        if export_pdb:
            cmd += f' && ambpdb -p {prmtop} -c {outputs["r"]} > {pdb}'
        return cmd

    def create(self, use_relpath: bool = True, relative_to: os.PathLike | None = None, export_pdb = True):
        self.wdir.mkdir(exist_ok=True)
        self.write_input()
        cmd = self.create_cmd(use_relpath, relative_to, export_pdb)
        with open(self.dirname / f'{self.name}.sh', 'w') as f:
            f.write('\n'.join(cmd.split('&&')))

    def link_prev_step(self, step):
        self.prev_step = step
        assert self.prev_step.prmtop == self.prmtop, "Not the same topology"
        self.set_inpcrd(self.prev_step.outputs['r'])


class Workflow:
    def __init__(self, wdir: os.PathLike, prmtop: os.PathLike, inpcrd: os.PathLike, steps: List[Step]):
        self.wdir = Path(wdir).resolve()
        self.steps = OrderedDict()
        steps[0].set_inpcrd(inpcrd)
        for i, step in enumerate(steps):
            step.set_prmtop(prmtop)
            step.set_parent_dir(self.wdir)
            self.steps[step.name] = step
            if i > 0:
                step.link_prev_step(steps[i - 1])
    
    def create(self, **kwargs):
        self.wdir.mkdir(True)
        for name, step in self.steps.items():
            step.create(**kwargs)


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
    

if __name__ == '__main__':
    print(write_cntrl({'imin': 1, 'dx0': 0.01, 'restaintmask': '!:WAT,Cl-,K+,Na+,NA,CL & !@H='}))