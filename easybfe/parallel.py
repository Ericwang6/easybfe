import sys
from typing import Callable, Any, List
import multiprocessing as mp
from tqdm import tqdm


def _determine_nprocs(nprocs):
    try:
        nprocs = int(nprocs)
    except:
        nprocs = -1
    
    nprocs = mp.cpu_count() - 2 if nprocs <= 0 else nprocs
    return max(1, nprocs)


def run_func_parallel(
    func: Callable, args_list: List[Any], 
    nprocs: int = -1, 
    unpack_args: bool = False, 
    progress_bar: bool = True,
    keep_order: bool = False,
    desc: str | None = None
):

    nprocs = _determine_nprocs(nprocs) if len(args_list) > 1 else 1
    progress_bar = progress_bar and len(args_list) > 1
    
    if nprocs == 1:
        iterator = tqdm(args_list, desc=desc, total=len(args_list)) if progress_bar else args_list
        results = []
        for arg in iterator:
            if unpack_args:
                r = func(*arg)
            else:
                r = func(arg)
            results.append(r)
    else:
        with mp.Pool(nprocs) as pool:
            if unpack_args:
                results = pool.starmap(func, args_list)
            elif keep_order:
                iterator = tqdm(pool.imap(func, args_list), desc=desc, total=len(args_list)) if progress_bar else pool.imap(func, args_list)
                results = [r for r in iterator]
            else:
                iterator = tqdm(pool.imap_unordered(func, args_list), desc=desc, total=len(args_list)) if progress_bar else pool.imap_unordered(func, args_list)
                results = [r for r in iterator]
    return results


            

