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


def process_file(f, output_folder, fps=30, width=-1, height=-1, quality=3, sync=False):
    # Setting both width and height != -1 does not preserve aspect ratio
    fname = os.path.basename(f)
    fname = os.path.splitext(fname)[0]
    safe_mkdirs(os.path.join(output_folder, fname))
    if width is not None or height is not None:
        out = run_cmd_silent(f"ffmpeg -y -i {f} -vf \"scale={width}:{height},fps={fps}\" -qscale:v {quality} -qmin 1 {output_folder}/{fname}/f_%04d.jpg", sync=sync)
    else:
        out = run_cmd_silent(f"ffmpeg -y -i {f} -vf \"fps={fps}\" -qscale:v {quality} -qmin 1 {output_folder}/{fname}/f_%04d.jpg", sync=sync)
    return out


def parse_args():
    parser = ArgumentParser("Extract frames from videos")
    parser.add_argument("-i", "--file-list", type=str, help="Text file with paths to input videos, one video per line")
    parser.add_argument("-o", "--out", type=str, help="Output folder to save frames. Will create if does not exist")
    parser.add_argument("--fps", type=int, default=30, help="FPS to sample video")
    parser.add_argument("-ww", "--width", type=int, default=-1, help="Width of final frames. If both width and height are set, aspect ratio is not preserved")
    parser.add_argument("-hh", "--height", type=int, default=-1, help="Height of final frames. If both width and height are set, aspect ratio is not preserved")
    parser.add_argument("-j", "--n-jobs", type=int, default=1, help="Num of jobs to use")
    parser.add_argument("-q", "--quality", type=int, default=3, help="JPEG quality. 1-31. Lower is better quality / more disc space")

    args = parser.parse_args()

    safe_mkdirs(args.out)

    if args.height > 0 and args.width > 0:
        print("WARNING: You have set width and height of output frames. Will not preserve aspect ratio")

    return args


if __name__ == "__main__":

    args = parse_args()


    flist = read_file_list(args.file_list)

    processes = []
    for f in tqdm(flist, desc="extracting wavs"):
        if args.n_jobs == 1:
            process_file(f, args.out, fps=args.fps, width=args.width, height=args.height, quality=args.quality, sync=True)
        else:
            pipe = process_file(f, args.out, fps=args.fps, width=args.width, height=args.height, quality=args.quality, sync=False)
            processes.append(pipe)
            if len(processes) == args.n_jobs:   
                for p in processes:
                    p.wait()
                processes = []

    for p in processes:
        p.wait()
