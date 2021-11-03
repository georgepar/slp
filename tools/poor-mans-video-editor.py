"""
Input: tsv file in the form

Input Video filename | topic | subtopic | title greek | title english | start time | end time | delete segments
input.mp4            | 1     | 1        | έξοδος      | output        | 00:10:05   | 00:30:10 | 00:11:15-00:12:30,00:20:35-00:22:10
"""

import os
import subprocess
import sys

import yaml


def run_cmd(command: str):
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
    print(stdout)

    return returncode, stdout


def out_video(segment, greek=True):
    title_idx = 3 if greek else 4
    title, topic, subtopic = segment[title_idx], segment[1], segment[2]
    name = f"{title}_{topic}-{subtopic}.mp4"

    return name


def input_video(segment):
    return segment[0]


def manage_timestamps(segment):
    try:
        st, et = segment[5], segment[6]
    except:
        st = segment[5]

        return [st]
    try:
        delete_timestamps = segment[7]
    except:
        return [st, et]

    if not delete_timestamps:
        return [st, et]
    else:
        return (
            [st]
            + [
                t

                for s in delete_timestamps.split(",")

                for t in (s.split("-")[0], s.split("-")[1])
            ]
            + [et]
        )


def to_cut_fmt(timestamp):
    out = ""
    labels = ["h", "m", "s"]
    lb_idx = 0

    for c in timestamp:
        if c == ":":
            out += labels[lb_idx]
            lb_idx += 1
        else:
            out += c

    return out


def to_cut_yaml(inmp4, outmp4, ymlname, timestamps):
    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)

        return list(zip(a, a))

    timestamps = [to_cut_fmt(t) for t in timestamps]
    timeframe = []

    if len(timestamps) == 1:
        timeframe = [{"from": "start", "to": timestamps[0]}]
    else:
        for s, e in pairwise(["start"] + timestamps + ["end"]):
            timeframe += [{"from": s, "to": e}]
    out = {
        "input": inmp4,
        "output": outmp4,
        "cut_method": "delete",
        "timeframe": timeframe,
    }
    with open(ymlname, "w") as fd:
        yaml.dump(out, fd, default_flow_style=False, sort_keys=False)


def format_timestamp_args(timestamps):
    if len(timestamps) == 1:
        return [f"-ss {timestamps[0]} "]

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)

        return list(zip(a, a))

    cmds = [f"-ss {s} -to {e}" for s, e in pairwise(timestamps)]

    return cmds


def ffmpeg(inp, out, timestamps_args):
    if len(timestamps_args) == 1:
        run_cmd(f"ffmpeg -y -i '{inp}' " + timestamps_args[0] + f" -c:v h265_nvenc -crf 24 -preset fast -c:a copy '{out}'")

        return
    mp4s = []

    for i, arg in enumerate(timestamps_args):
        mp4s.append(f"{i}.mp4")
        cmd = f"ffmpeg  -i '{inp}' " + arg + f" -c:v h265_nvenc -crf 24 -preset fast -c:a copy '{i}.mp4'"
        print(cmd)
        run_cmd(cmd)

    tmp = ".tmp_files.txt"
    with open(tmp, "w") as fd:
        for f in mp4s:
            fd.write(f"file '{f}'\n")

    run_cmd(f"ffmpeg -y -f concat -i .tmp_files.txt '{out}'")
    run_cmd(f"rm {tmp} " + " ".join(mp4s))


def read_split_tsv(timestamp_file):
    with open(timestamp_file) as f:
        segments = [ln.strip().split("\t") for ln in f]

    return segments


def main():
    timestamp_file = sys.argv[1]
    segments = read_split_tsv(timestamp_file)

    for segment in segments:
        inmp4 = input_video(segment)
        outmp4 = "out/" + out_video(segment, greek=True)
        timestamps = manage_timestamps(segment)
        timestamp_args = format_timestamp_args(timestamps)
        ffmpeg(inmp4, outmp4, timestamp_args)


def main1():
    timestamp_file = sys.argv[1]
    segments = read_split_tsv(timestamp_file)

    for i, segment in enumerate(segments):
        inmp4 = input_video(segment)
        outmp4 = out_video(segment, greek=True)
        timestamps = manage_timestamps(segment)
        to_cut_yaml(inmp4, outmp4, f"{i}.yml", timestamps)


if __name__ == "__main__":
    main()
