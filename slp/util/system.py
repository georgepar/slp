import functools
import os
import pickle
import shutil
import subprocess
import sys
import time
import urllib
import urllib.request
import validators

from typing import cast, Any, Callable, Optional, Tuple

from slp.util import log
from slp.util import types

ERROR_INVALID_NAME: int = 123


try:
    import ujson as json
except ImportError:
    import json  # type: ignore


def print_separator(symbol: str = '*',
                    n: int = 10,
                    print_fn: Callable[[str], None] = print):
    print_fn(symbol * n)


def is_url(inp: Optional[str]) -> types.ValidationResult:
    if not inp:
        return False
    return validators.url(inp)


def is_file(inp: Optional[str]) -> types.ValidationResult:
    if not inp:
        return False
    return os.path.isfile(inp)


def is_subpath(child: str, parent: str) -> bool:
    parent = os.path.abspath(parent)
    child = os.path.abspath(child)
    return cast(bool,
                os.path.commonpath([parent]) ==
                os.path.commonpath([parent, child]))


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            log.warning(e)
            raise IOError(
                (f"Failed to create recursive directories: {path}"))


def timethis(func: Callable) -> Callable:
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @slp.util.sys.timethis
      def time_consuming_function(...):
    """
    @functools.wraps(func)
    def timed(*args: types.T, **kwargs: types.T):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = f'{te - ts}'
        log.info(
            'BENCHMARK: {f}(*{a}, **{kw}) took: {t} sec'.format(
                f=func.__name__, a=args, kw=kwargs, t=elapsed))
        return result
    return cast(Callable, timed)


def suppress_print(func: Callable) -> Callable:
    def func_wrapper(*args: types.T, **kwargs: types.T):
        with open('/dev/null', 'w') as sys.stdout:
            ret = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return ret
    return cast(Callable, func_wrapper)


def run_cmd(command: str) -> Tuple[int, str]:
    """
    Run given command locally
    Return a tuple with the return code, stdout, and stderr of the command
    """
    command = f'{os.getenv("SHELL")} -c "{command}"'
    pipe = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    stdout = ''.join([line.decode("utf-8")
                      for line in iter(pipe.stdout.readline, b'')])
    pipe.stdout.close()
    returncode = pipe.wait()
    return returncode, stdout


def run_cmd_silent(command: str) -> Tuple[int, str]:
    return cast(Tuple[int, str], suppress_print(run_cmd)(command))


def download_url(url: str, dest_path: str) -> str:
    """
    Download a file to a destination path given a URL
    """
    name = url.rsplit('/')[-1]
    dest = os.path.join(dest_path, name)
    safe_mkdirs(dest_path)
    response = urllib.request.urlopen(url)
    with open(dest, 'wb') as fd:
        shutil.copyfileobj(response, fd)
    return dest


def write_wav(byte_str: str, wav_file: str) -> None:
    '''
    Write a hex string into a wav file

    Args:
        byte_str: The hex string containing the audio data
        wav_file: The output wav file

    Returns:
    '''
    with open(wav_file, 'w') as fd:
        fd.write(byte_str)


def read_wav(wav_sample: str) -> str:
    '''
    Reads a wav clip into a string
    and returns the hex string.
    Args:

    Returns:
        A hex string with the audio information.
    '''
    with open(wav_sample, 'r') as wav_fd:
        clip = wav_fd.read()
    return clip


def pickle_load(fname: str) -> Any:
    with open(fname, 'rb') as fd:
        data = pickle.load(fd)
    return data


def pickle_dump(data: Any, fname: str) -> None:
    with open(fname, 'wb') as fd:
        pickle.dump(data, fd)


def json_load(fname: str) -> types.GenericDict:
    with open(fname, 'r') as fd:
        data = json.load(fd)
    return cast(types.GenericDict, data)


def json_dump(data: types.GenericDict, fname: str) -> None:
    with open(fname, 'w') as fd:
        json.dump(data, fd)
