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
import socket

from loguru import logger
from typing import cast, Any, Callable, Optional, Tuple

from datetime import datetime
from slp.util import types

try:
    import ujson as json
except ImportError:
    import json  # type: ignore


def has_internet_connection(timeout: int = 3) -> bool:
    """has_internet_connection Check if you are connected to the internet

    Check if internet connection exists by pinging Google DNS server

    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)

    Args:
        timeout (int): Seconds to wait before giving up

    Returns:
        bool: True if connection is established, False if we are not connected to the internet
    """
    host, port = "8.8.8.8", 53
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def date_fname() -> str:
    """date_fname Generate a filename based on datetime.now().

    If multiple calls are made within the same second, the filename will not be unique.
    We could add miliseconds etc. in the fname but that would hinder readability.
    For practical purposes e.g. unique logs between different experiments this should be enough.
    Either way if we need a truly unique descriptor, there is the uuid module.

    Returns:
        str: A filename, e.g. 20210228-211832
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_separator(
    symbol: str = "*", n: int = 10, print_fn: Callable[[str], None] = print
):
    """print_separator Print a repeated symbol as a separator

    *********************************************************

    Args:
        symbol (str): Symbol to print
        n (int): Number of times to print the symbol
        print_fn (Callable[[str], None]): Print function to use, e.g. print or logger.info

    Examples:
        >>> print_separator(symbol="-", n=2)
        --
    """
    print_fn(symbol * n)


def is_url(inp: Optional[str]) -> types.ValidationResult:
    """is_url Check if the provided string is a URL

    Args:
        inp (Optional[str]): A potential link or None

    Returns:
        types.ValidationResult: True if a valid url is provided, False if the string is not a url

    Examples:
        >>> is_url("Hello World")
        ValidationFailure(func=url, args={'value': 'Hello World', 'public': False})
        >>> is_url("http://google.com")
        True
    """
    if not inp:
        return False
    return validators.url(inp)


def is_file(inp: Optional[str]) -> types.ValidationResult:
    """is_file Check if the provided string is valid file in the system path

    Args:
        inp (Optional[str]): A potential file or None

    Returns:
        types.ValidationResult: True if a valid file is provided, False if the string is not a url

    Examples:
        >>> is_file("/bin/bash")
        True
        >>> is_file("/supercalifragilisticexpialidocious")  # This does not exist. I hope...
        False
    """
    if not inp:
        return False
    return os.path.isfile(inp)


def is_subpath(child: str, parent: str) -> bool:
    """is_subpath Check if child path is a subpath of parent

    Args:
        child (str): Child path
        parent (str): parent path

    Returns:
        bool: True if child is a subpath of parent, false if not

    Examples:
        >>> is_subpath("/usr/bin/Xorg", "/usr")
        True
    """
    parent = os.path.abspath(parent)
    child = os.path.abspath(child)
    return cast(
        bool, os.path.commonpath([parent]) == os.path.commonpath([parent, child])
    )


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
            logger.warning(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def timethis(method=False) -> Callable:
    """timethis Decorator to measure the time it takes for a function to complete

    Examples:
        >>> @slp.util.system.timethis
        >>> def time_consuming_function(...): ...
    """

    def timethis_inner(func: Callable) -> Callable:
        """Inner function for decorator closure"""

        @functools.wraps(func)
        def timed(*args: types.T, **kwargs: types.T):
            """Inner function for decorator closure"""

            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            elapsed = f"{te - ts}"
            if method:

                logger.info(
                    "BENCHMARK: {cls}.{f}(*{a}, **{kw}) took: {t} sec".format(
                        f=func.__name__, cls=args[0], a=args[1:], kw=kwargs, t=elapsed
                    )
                )
            else:
                logger.info(
                    "BENCHMARK: {f}(*{a}, **{kw}) took: {t} sec".format(
                        f=func.__name__, a=args, kw=kwargs, t=elapsed
                    )
                )
            return result

        return cast(Callable, timed)

    return timethis_inner


def suppress_print(func: Callable) -> Callable:
    """suppress_print Decorator to supress stdout of decorated function

    Examples:
        >>> @slp.util.system.timethis
        >>> def very_verbose_function(...): ...
    """

    def func_wrapper(*args: types.T, **kwargs: types.T):
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


def run_cmd_silent(command: str) -> Tuple[int, str]:
    """run_cmd_silent Run command without printing to console

    Args:
        command (str): Shell command to run

    Returns:
        (int, str): Status code, stdout of shell command

    Examples:
        >>> run_cmd("ls /")
        (0, 'bin\nboot\ndev\netc\nhome\ninit\nlib\nlib32\nlib64\nlibx32\nlost+found\nmedia\nmnt\nopt\nproc\nroot\nrun\nsbin\nsnap\nsrv\nsys\ntmp\nusr\nvar\n')
    """
    return cast(Tuple[int, str], suppress_print(run_cmd)(command))


def download_url(url: str, dest_path: str) -> str:
    """download_url Download a file to a destination path given a URL

    Args:
        url (str): A url pointing to the file we want to download
        dest_path (str): The destination path to write the file

    Returns:
        (str): The filename where the downloaded file is written
    """
    name = url.rsplit("/")[-1]
    dest = os.path.join(dest_path, name)
    safe_mkdirs(dest_path)
    response = urllib.request.urlopen(url)
    with open(dest, "wb") as fd:
        shutil.copyfileobj(response, fd)
    return dest


def write_wav(byte_str: str, wav_file: str) -> None:
    """write_wav Write a hex string into a wav file

    Args:
        byte_str (str): The hex string containing the audio data
        wav_file (str): The output wav file
    """
    with open(wav_file, "w") as fd:
        fd.write(byte_str)


def read_wav(wav_sample: str) -> str:
    """read_wav Reads a wav clip into a string and returns the hex string.

    Args:
        wav_sample (str): Path to wav file

    Returns:
        A hex string with the audio information.
    """
    with open(wav_sample, "r") as wav_fd:
        clip = wav_fd.read()
    return clip


def pickle_load(fname: str) -> Any:
    """pickle_load Load data from pickle file

    Args:
        fname (str): file name of pickle file

    Returns:
        Any: Loaded data
    """
    with open(fname, "rb") as fd:
        data = pickle.load(fd)
    return data


def pickle_dump(data: Any, fname: str) -> None:
    """pickle_dump Save data to pickle file

    Args:
        data (Any): Data to save
        fname (str): Output pickle file
    """
    with open(fname, "wb") as fd:
        pickle.dump(data, fd)


def json_load(fname: str) -> types.GenericDict:
    """json_load Load dict from a json file

    Args:
        fname (str): Json file to load

    Returns:
        types.GenericDict: Dict of loaded data
    """
    with open(fname, "r") as fd:
        data = json.load(fd)
    return cast(types.GenericDict, data)


def json_dump(data: types.GenericDict, fname: str) -> None:
    """json_dump Save dict to a json file

    Args:
        data (types.GenericDict): Dict to save
        fname (str): Output json file
    """
    with open(fname, "w") as fd:
        json.dump(data, fd)
