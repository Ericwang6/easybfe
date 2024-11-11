import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import shutil
import math


run_sh = '''
prmtop="{prmtop}"
inpcrd="{inpcrd}"
deffnm="{deffnm}"
{pmemd_exec} -O \
-i $deffnm.in -o $deffnm.out -p $prmtop -c $inpcrd \
-r $deffnm.rst7 -inf $deffnm.info -ref $inpcrd \
-x $deffnm.mdcrd -e $deffnm.mden -l $deffnm.log
ambpdb -p $prmtop -c $deffnm.rst7 > $deffnm.pdb
'''

def _fe_var_check(var, varname):
    msg = f"{varname} has to be set when run free energy simulation"
    if isinstance(var, str):
        assert var != "", msg
    else:
        assert (var is not None), msg


def pmemd_exec(use_cuda: bool = True, use_mpi: bool = False):
    if use_cuda:
        exe = "pmemd.cuda"
    else:
        exe = "pmemd"
    if use_mpi:
        exe += '.MPI'
    return exe


def pmemd_command(pmemed_exec, prmtop, inpcrd, deffnm):
    return run_sh.format(prmtop=prmtop, inpcrd=inpcrd, deffnm=deffnm, pmemd_exec=pmemed_exec)


def create_heating_schedule(nsteps: int, final_temp: float = 298.15, init_temp: float = 5.0):
    start = math.ceil(init_temp / 100)
    end = math.ceil(final_temp / 100)

    schedule = [init_temp]
    for i in range(start, end):
        schedule.append(i * 100)
    schedule.append(final_temp)

    nsteps_stage = math.floor(nsteps /(len(schedule) - 1) / 2)

    lines = []
    for i in range(len(schedule) - 1):
        lines.append(
            f'&wt TYPE="TEMP0", istep1 = {i * 2 * nsteps_stage + 1}, istep2 = {(i * 2 + 1) * nsteps_stage}, value1 = {schedule[i]}, value2 = {schedule[i+1]}, /'
        )
        lines.append(
            f'&wt TYPE="TEMP0", istep1 = {(i * 2 + 1) * nsteps_stage + 1}, istep2 = {(i + 1) * 2 * nsteps_stage}, value1 = {schedule[i+1]}, value2 = {schedule[i+1]}, /'
        )
    lines.append('&wt TYPE="END", /')
    lines.append("")

    return lines


def em(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = "em",
    use_periodic: bool = True,
    step_size: float = 0.01,
):
    """
    Energy minimization
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'em.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
    else:
        ifsc, icfe = 0, 0
        clambda = 0.0

    ntb = 1 if use_periodic else 0

    inpstr = template.format(
        maxcyc=num_steps, ofreq=ofreq,
        cut=cutoff, 
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2,
        ntb=ntb, step_size=step_size
    )
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(
        pmemd_exec, 
        os.path.relpath(prmtop, wdir), 
        os.path.relpath(inpcrd, wdir), 
        deffnm
    )
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def heat(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    tempi: float = 0.0,
    restraint_wt: float = 5.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'heat',
    use_periodic: bool = True,
    restart: bool = False,
    charge_change_mdin_mod: List[str] = list()
):
    """
    Heating the system (NVT equilibrium)
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'heat.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0
    ntb = 1 if use_periodic else 0
    iwrap = 0

    if restart:
        ntx, irest = 5, 1
    else:
        ntx, irest = 1, 0

    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        ntf=ntf,
        timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2,
        ntb=ntb, iwrap=iwrap,
        temp0=temp0, tempi=tempi,
        ntx=ntx, irest=irest
    )

    mdin_mod = create_heating_schedule(num_steps, temp0, tempi)
    if charge_change_mdin_mod:
        mdin_mod = mdin_mod[:-1] + charge_change_mdin_mod
    inpstr += '\n' + '\n'.join(mdin_mod)

    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(
        pmemd_exec, 
        os.path.relpath(prmtop, wdir), 
        os.path.relpath(inpcrd, wdir), 
        deffnm
    )
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def pressurize(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    pressure: float = 1.01325,
    restraint_wt: float = 5.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'pres_0',
    use_periodic: bool = True,
    charge_change_mdin_mod: List[str] = list(),
):
    """
    Pressurize the system (NPT equilibrium)
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'pres.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0
    ntb = 2 if use_periodic else 0
    iwrap = 0
    ntp = 1 if use_periodic else 0
    nmropt = 1 if charge_change_mdin_mod else 0

    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        temp0=temp0,
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        ntf=ntf,
        pres0=pressure,
        timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2,
        ntb=ntb, iwrap=iwrap, ntp=ntp,
        nmropt=nmropt
    )
    inpstr += '\n' + '\n'.join(charge_change_mdin_mod)
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(
        pmemd_exec, 
        os.path.relpath(prmtop, wdir), 
        os.path.relpath(inpcrd, wdir), 
        deffnm
    )
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def prod(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    pressure: float = 1.01325,
    restraint_wt: float = 0.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    use_mbar: bool = True,
    lambdas: Optional[List[float]] = None,
    efreq: Optional[int] = None,
    numexchg: Optional[int] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'prod',
    use_periodic: bool = True,
    use_hremd: bool = True,
    use_nvt: bool = False,
    charge_change_mdin_mod: List[str] = list()
):
    """
    Production run
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'prod.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0

    if use_periodic:
        ntb = 1 if use_nvt else 2
        ntp = 0 if use_nvt else 1
        iwrap = 1
    else:
        ntb = 0
        ntp = 0
        iwrap = 0

    if free_energy and use_mbar:
        _fe_var_check(lambdas, "lambdas")
        efreq = num_steps if efreq is None else efreq
        mbar_setting = [
            "{:<15} = 1".format("ifmbar"), 
            "{:<15} = {}".format("bar_intervall", efreq),
            "{:<15} = {}".format("mbar_states", len(lambdas)),
            "{:<15} = {}".format("mbar_lambda", ",".join(str(x) for x in lambdas))
        ]
        mbar_setting = "\n".join(mbar_setting)
    else:
        efreq = ofreq if efreq else efreq
        mbar_setting = ""
    
    if use_hremd:
        _fe_var_check(numexchg, "numexchg")
        remd_setting = [
            "{:<15} = {},".format("numexchg", numexchg),
            "{:<15} = {},".format("gremd_acyc", len(lambdas) % 2)
        ]
        remd_setting = '\n'.join(remd_setting)
    else:
        remd_setting = ""
    
    nmropt = 1 if charge_change_mdin_mod else 0
    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        temp0=temp0,
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        ntf=ntf,
        pres0=pressure,
        timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2,
        remd_setting=remd_setting, mbar_setting=mbar_setting,
        efreq=efreq,
        ntb=ntb, iwrap=iwrap, ntp=ntp,
        nmropt=nmropt
    )
    inpstr += '\n' + '\n'.join(charge_change_mdin_mod)
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    if not use_periodic:
        cmdstr = pmemd_command(
            pmemd_exec, 
            os.path.relpath(prmtop, wdir), 
            os.path.relpath(inpcrd, wdir), 
            deffnm
        )
    else:
        cmdstr = f"ambpdb -p {os.path.relpath(prmtop, wdir)} -c {deffnm}.rst7 > {deffnm}.pdb"
        
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def fep_workflow(config, wdir, gas_phase: bool = False, use_prev_lambda_as_start: bool = True, charge_change_mdin_mod: List[str] = list()):
    lambdas = config['lambdas']
    inpcrd = Path(config['inpcrd']).resolve()
    prmtop = Path(config['prmtop']).resolve()

    mask_config = {
        key: config[key] for key in ['timask1', 'timask2', 'scmask1', 'scmask2']
    }

    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    
    temp = config.get("temperature", 298.15)
    pres = config.get("pressure", 1.01325)
    cutoff = config.get("cutoff", 10.0)
    
    for i, clambda in enumerate(lambdas):
        lambda_dir = wdir / f"lambda{i}"
        lambda_dir.mkdir(exist_ok=True)
        
        em_dir = lambda_dir / "em"

        if use_prev_lambda_as_start and i > 0:
            em_inpcrd = wdir / f'lambda{i - 1}/em/em.rst7'
        else:
            em_inpcrd = inpcrd
        
        em(
            wdir=em_dir,
            prmtop=prmtop, 
            inpcrd=em_inpcrd,
            pmemd_exec=pmemd_exec(use_cuda=True, use_mpi=False),
            free_energy=True,
            cutoff=cutoff,
            clambda=clambda,
            deffnm="em",
            **config['em'],
            **mask_config,
        )

        tempi_list = [5.0, 100.0, 200.0]
        temp0_list = [100.0, 200.0, temp]
        if not gas_phase:
            for i in range(3):
                heat_dir = lambda_dir / f'heat_{i}'
                if i == 0:
                    restart = False
                    heat_inpcrd = em_dir / 'em.rst7' 
                else:
                    restart = True
                    heat_inpcrd = lambda_dir / f'pres_{i-1}/pres_{i-1}.rst7'
                
                heat(
                    wdir=heat_dir,
                    prmtop=prmtop,
                    inpcrd=heat_inpcrd,
                    cutoff=cutoff,
                    temp0=temp0_list[i],
                    tempi=tempi_list[i],
                    free_energy=True,
                    clambda=clambda,
                    restart=restart,
                    deffnm=f'heat_{i}',
                    charge_change_mdin_mod=charge_change_mdin_mod,
                    **config[f'heat_{i}'],
                    **mask_config
                )

                pres_dir = lambda_dir / f'pres_{i}'
                pressurize(
                    wdir=pres_dir,
                    prmtop=prmtop,
                    inpcrd=heat_dir / f"heat_{i}.rst7",
                    cutoff=cutoff,
                    pressure=pres,
                    temp0=temp0_list[i], 
                    free_energy=True, clambda=clambda,
                    charge_change_mdin_mod=charge_change_mdin_mod,
                    deffnm=f'pres_{i}',
                    **config[f'pres_{i}'],
                    **mask_config
                )

            pre_prod_dir = lambda_dir / "pre_prod"
            pressurize(
                wdir=pre_prod_dir,
                prmtop=prmtop,
                inpcrd=lambda_dir / 'pres_2/pres_2.rst7',
                cutoff=cutoff,
                pressure=pres,
                temp0=temp, 
                free_energy=True, clambda=clambda,
                deffnm='pre_prod',
                charge_change_mdin_mod=charge_change_mdin_mod,
                **config['pre_prod'],
                **mask_config
            )
            prod_inpcrd = pre_prod_dir / 'pre_prod.rst7'
        else:
            heat_dir = lambda_dir / 'heat'
            heat(
                wdir=heat_dir,
                prmtop=prmtop,
                inpcrd=lambda_dir / 'em/em.rst7',
                cutoff=cutoff,
                temp0=temp, 
                free_energy=True,
                clambda=clambda,
                restart=False,
                deffnm='heat',
                **config['heat'],
                **mask_config
            )
            prod_inpcrd = heat_dir / 'heat.rst7'

        prod_dir = lambda_dir / "prod"
        prod(
            wdir=prod_dir,
            prmtop=prmtop,
            inpcrd=prod_inpcrd,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            restraint_wt=0.0,
            use_mbar=True,
            deffnm='prod',
            lambdas=lambdas,
            use_nvt=gas_phase,
            charge_change_mdin_mod=charge_change_mdin_mod,
            **config['prod'],
            **mask_config
        )


    if gas_phase:
        stages = ['em', 'heat', 'prod']
    else:
        stages = ['em', 'heat_0', 'pres_0', 'heat_1', 'pres_1', 'heat_2', 'pres_2', 'pre_prod', 'prod']

    
    for si in range(1, len(stages)):
        stage = stages[si]
        prev_stage = stages[si-1]

        groupfile = []
        str_template = "-O -p {prmtop} -c {inpcrd} -i {mdin} -o {mdout} -r {restart} -x {traj} -ref {ref} -e {mden} -l {mdlog} -inf {mdinfo}"
        for i in range(len(lambdas)):
            groupfile.append(str_template.format(
                prmtop=os.path.relpath(prmtop, wdir),
                inpcrd=f"lambda{i}/{prev_stage}/{prev_stage}.rst7",
                mdin=f"lambda{i}/{stage}/{stage}.in",
                mdout=f"lambda{i}/{stage}/{stage}.out",
                restart=f"lambda{i}/{stage}/{stage}.rst7",
                traj=f"lambda{i}/{stage}/{stage}.mdcrd",
                ref=f"lambda{i}/{prev_stage}/{prev_stage}.rst7",
                mden=f"lambda{i}/{stage}/{stage}.mden",
                mdlog=f"lambda{i}/{stage}/{stage}.log",
                mdinfo=f"lambda{i}/{stage}/{stage}.info"
            ))
        with open(wdir / f"{stage}.groupfile", 'w') as f:
            f.write('\n'.join(groupfile))
    
    with open(Path(__file__).parent / 'run.sh.template') as f:
        script = f.read()
    
    script = script.replace('@NUM_LAMBDA', str(len(lambdas)))
    script = script.replace('@STAGES', '({})'.format(' '.join(f'"{stage}"' for stage in stages)))
    script = script.replace('@MD_EXEC', pmemd_exec(True, True))

    with open(wdir / 'run.sh', 'w') as f:
        f.write(script)