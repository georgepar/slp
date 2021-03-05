#!/usr/bin/env python

import sys
import os
import subprocess
from typing import Callable, Tuple, cast
from tqdm import tqdm
from argparse import ArgumentParser



def safe_mkdirs(path: str) -> None:
    """safe_mkdirs Makes recursively all the directories in input path
    Utility function similar to mkdir -p. Makes directories recursively, if given path does not exist
    Args:
        path (str): Path to mkdir -p
    Examples:
        >>> safe_mkdirs("super/cali/fragi/listic/expi/ali/docious")
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            raise IOError((f"Failed to create recursive directories: {path}"))


def suppress_print(func: Callable) -> Callable:
    """suppress_print Decorator to supress stdout of decorated function
    Examples:
        >>> @slp.util.system.timethis
        >>> def very_verbose_function(...): ...
    """

    def func_wrapper(*args, **kwargs):
        """Inner function for decorator closure"""
        with open("/dev/null", "w") as sys.stdout:
            ret = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return ret

    return cast(Callable, func_wrapper)


def run_cmd(command: str) -> Tuple[int, str]:
    """run_cmd Run given shell command
    Args:
        command (str): Shell command to run
    Returns:
        (int, str): Status code, stdout of shell command
    Examples:
        >>> run_cmd("ls /")
        (0, 'bin\nboot\ndev\netc\nhome\ninit\nlib\nlib32\nlib64\nlibx32\nlost+found\nmedia\nmnt\nopt\nproc\nroot\nrun\nsbin\nsnap\nsrv\nsys\ntmp\nusr\nvar\n')
    """
    command = f'{os.getenv("SHELL")} -c "{command}"'
    pipe = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout = ""
    if pipe.stdout is not None:
        stdout = "".join(
            [line.decode("utf-8") for line in iter(pipe.stdout.readline, b"")]
        )
        pipe.stdout.close()
    returncode = pipe.wait()
    return returncode, stdout


def run_cmd_async(command: str) -> Tuple[int, str]:
    """run_cmd Run given shell command
    Args:
        command (str): Shell command to run
    Returns:
        (int, str): Status code, stdout of shell command
    Examples:
        >>> run_cmd("ls /")
        (0, 'bin\nboot\ndev\netc\nhome\ninit\nlib\nlib32\nlib64\nlibx32\nlost+found\nmedia\nmnt\nopt\nproc\nroot\nrun\nsbin\nsnap\nsrv\nsys\ntmp\nusr\nvar\n')
    """
    command = f'{os.getenv("SHELL")} -c "{command}"'
    pipe = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    return pipe


def run_cmd_silent(command: str, sync=False) -> Tuple[int, str]:
    """run_cmd_silent Run command without printing to console
    Args:
        command (str): Shell command to run
    Returns:
        (int, str): Status code, stdout of shell command
    Examples:
        >>> run_cmd("ls /")
        (0, 'bin\nboot\ndev\netc\nhome\ninit\nlib\nlib32\nlib64\nlibx32\nlost+found\nmedia\nmnt\nopt\nproc\nroot\nrun\nsbin\nsnap\nsrv\nsys\ntmp\nusr\nvar\n')
    """
    cmd = run_cmd if sync else run_cmd_async
    return suppress_print(cmd)(command)


def read_file_list(flist):
    with open(flist, "r") as fd:
        files = [f.strip() for f in fd]
    return files


def process_file(f, output_folder, sync=False):
    # Setting both width and height != -1 does not preserve aspect ratio
    fname = os.path.basename(f)
    fname = os.path.splitext(fname)[0]
    out = run_cmd_silent(f"ffmpeg -y -i {f} -vn -acodec copy {output_folder}/{fname}.wav", sync=sync)
    return out


def parse_args():
    parser = ArgumentParser("Extract wavs from videos")
    parser.add_argument("-i", "--file-list", type=str, help="Text file with paths to input videos, one video per line")
    parser.add_argument("-o", "--out", type=str, help="Output folder to save wavs. Will create if does not exist")
    parser.add_argument("-j", "--n-jobs", type=int, default=1, help="Num of jobs to use")

    args = parser.parse_args()

    safe_mkdirs(args.out)

    return args


if __name__ == "__main__":

    args = parse_args()

    flist = read_file_list(args.file_list)

    processes = []
    for f in tqdm(flist, desc="extracting wavs"):
        if args.n_jobs == 1:
            process_file(f, args.out, sync=True)
        else:
            pipe = process_file(f, args.out, sync=False)
            processes.append(pipe)
            if len(processes) == args.n_jobs:   
                for p in processes:
                    p.wait()
                processes = []


