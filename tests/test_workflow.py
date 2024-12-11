from pathlib import Path
import shutil
from easybfe.amber.workflow import Step, Workflow


def create_test_file(name):
    path = Path(__file__).resolve().parent / ('_test_' + name)
    fp = open(path, 'w')
    fp.write('This file is a test file')
    fp.close()
    return path


def test_step():
    prmtop = create_test_file('test.prmtop')
    inpcrd = create_test_file('test.inpcrd')
    mdin = create_test_file('test.in')
    step = Step(
        name='test',
        wdir=Path(__file__).resolve().parent / '_test_step',
        mdin=mdin,
        prmtop=prmtop,
        inpcrd=inpcrd
    )
    step.create(use_relpath=True)
    prmtop.unlink()
    inpcrd.unlink()
    mdin.unlink()


def test_workflow():
    wdir = Path(__file__).resolve().parent / '_test_workflow'
    prmtop = create_test_file('wf.prmtop')
    inpcrd = create_test_file('wf.inpcrd')
    mdins = []
    steps = []
    for i in range(3):
        mdin = create_test_file(f'step_{i}.in')
        mdins.append(mdin)
        step = Step(name=f'step_{i}', mdin=mdin)
        steps.append(step)
    wf = Workflow(
        wdir=wdir,
        prmtop=prmtop,
        inpcrd=inpcrd,
        steps=steps
    )
    wf.create()
    shutil.rmtree(wdir)
    prmtop.unlink()
    inpcrd.unlink()
    for mdin in mdins:
        mdin.unlink()

