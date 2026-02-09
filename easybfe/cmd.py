"""
Author: Eric Wang
Date: 10/19/2022

This package contains useful functions to manipulate command line executions.
"""

import os
import sys
import io
import subprocess
import shutil
from pathlib import Path
from typing import List, Union, Optional, Tuple, Any
import contextlib
import logging
import uuid


@contextlib.contextmanager
def get_text_stream(src: Union[str, bytes, Path, io.IOBase], encoding: Optional[str] = None):
    """
    Convert any input to text stream
    """
    # Case 1: already io
    if hasattr(src, "read"):
        if isinstance(src, (io.BufferedIOBase, io.RawIOBase)) or "b" in getattr(src, "mode", ""):
            wrapper = io.TextIOWrapper(src, encoding=encoding)
            try:
                if hasattr(src, "seek"): src.seek(0)
                yield wrapper
            finally:
                wrapper.detach()
        else:
            yield src

    # Case 2: bytes
    elif isinstance(src, bytes):
        yield io.TextIOWrapper(io.BytesIO(src), encoding=encoding)

    # Case 3: string or pathlike
    elif isinstance(src, str):
        # check if it's a path or content
        if (len(src) < 1024) and ("\n" not in src) and os.path.isfile(src):
            with open(src, "r", encoding=encoding) as f:
                yield f
        else:
            yield io.StringIO(src)
    elif isinstance(src, Path):
        with open(src, 'r', encoding=encoding) as fp:
            yield fp
    else:
        raise TypeError(f"Unsupported source type: {type(src)}")



def smart_copy(src: Any, dst: str | Path, allow_formats: Optional[list[str]] = None):
    if isinstance(src, bytes):
        Path(dst).write_bytes(src)
    elif hasattr(src, "read"):
        with open(dst, 'wb'):
            shutil.copyfileobj(src, dst)
    elif os.path.isfile(src):
        fmt = Path(src).suffix[1:]
        if allow_formats and fmt not in allow_formats:
            raise TypeError(f"Not allowed format: {fmt} (allowed {allow_formats})")
        shutil.copyfile(src, dst)
    elif isinstance(src, str) or isinstance(src, Path):
        raise FileNotFoundError(src)
    else:
        raise TypeError(f'Unrecognized type: {type(src)}')


def init_logger(logname: Optional[os.PathLike] = None) -> logging.Logger:
    # logging
    logger = logging.getLogger('easybfe')
    logger.propagate = False
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file
    if logname is not None:
        handler = logging.FileHandler(str(logname))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


class CommandExecuteError(Exception):
    """
    Exception for command line exec error
    """
    def __init__(self, msg: str):
        self.msg = msg
    
    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg


class ExecutableNotFoundError(Exception):
    """
    Exception for not found executable
    """
    def __init__(self, exec: str):
        self.exec = exec
        self.msg = f"{self.exec} not found"
    
    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg


def find_executable(execs: Union[str, List[str]]) -> str:
    """
    Find executables

    Parameters
    ----------
    exec: str or list
        Executable to find. If a list provided, will return the first existing executable.
    
    Raises
    ------
    ExecutableNotFoundError:
        Raises if not find executables
    
    Return
    ------
    found_exec: str
        executable found
    """
    execs = execs if isinstance(execs, list) else [execs]

    found_exec = None
    for e in execs:
        found_exec = shutil.which(e)
        if found_exec is not None:
            break
    if found_exec is None:
        raise ExecutableNotFoundError(" or ".join(execs))
    else:
        return found_exec


def run_command(
    cmd: Union[List[str], str],
    raise_error: bool = True,
    input: Optional[str] = None,
    timeout: Optional[int] = None,
    cwd: str = '.',
    **kwargs,
) -> Tuple[int, str, str]:
    """
    Run shell command in subprocess

    Parameters
    ----------
    cmd: list of str, or str
        Command to execute
    raise_error: bool
        Whether to raise an error if the command failed
    input: str, optional
        Input string for the command
    timeout: int, optional
        Timeout for the command
    cwd: str
        Directory to run the command. Default is '.'
    **kwargs:
        Arguments in subprocess.Popen
    
    Raises
    ------
    AssertionError:
        Raises if the error failed to execute and `raise_error` set to `True`
    
    Return
    ------
    return_code: int
        The return code of the command
    out: str
        stdout content of the executed command
    err: str
        stderr content of the executed command
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]

    logger = kwargs.pop('logger', None)
    if logger:
        logger.info(f'The following command is running at: {Path(cwd).resolve()}:\n{" ".join(cmd)}')

    sub = subprocess.Popen(
        args=cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        **kwargs
    )
    if input is not None:
        sub.stdin.write(bytes(input, encoding=sys.stdin.encoding))
    try:
        out, err = sub.communicate(timeout=timeout)
        return_code = sub.poll()
    except subprocess.TimeoutExpired:
        sub.kill()
        print("Command %s timeout after %d seconds" % (cmd, timeout))
        return 999, "", ""  # 999 is a special return code for timeout
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if raise_error and return_code != 0:
        cmdstr = " ".join(cmd) if isinstance(cmd, list) else cmd
        raise CommandExecuteError("Command %s failed: \n%s" % (cmdstr, err))
    return return_code, out, err


@contextlib.contextmanager
def set_directory(dirname: os.PathLike, mkdir: bool = False):
    """
    Set current workding directory within context
    
    Parameters
    ----------
    dirname : os.PathLike
        The directory path to change to
    mkdir: bool
        Whether make directory if `dirname` does not exist
    
    Yields
    ------
    path: Path
        The absolute path of the changed working directory
    
    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)
