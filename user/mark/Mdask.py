#!/usr/bin/env python3
"""Distribute an embarrassingly parallel Madagascar pipeline with Dask.

SFDASK is a Dask Distributed analogue of sfmpi.  It windows an RSF or MDIO
input along one axis, runs the complete command after ``--`` for every window,
and joins the results (or adds them when join=0).

Examples
--------

  sfdask filein=in.rsf fileout=out.rsf split=3 -- sfscale dscale=2

  sfdask filein=in.mdio informat=mdio fileout=out.mdio outformat=mdio \
    split=3 -- 'sfscale dscale=2 | sfbandpass'

An unquoted shell pipe is interpreted by the invoking shell, not by sfdask.
External workers must see the same files and Madagascar executables.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class DaskRsfError(RuntimeError):
    """A user-facing sfdask failure."""


@dataclass(frozen=True)
class Partition:
    index: int
    start: int
    count: int


@dataclass
class ParsedCommand:
    options: Dict[str, str]
    aux_inputs: Dict[str, str]
    aux_outputs: Dict[str, str]
    command: str


_WRAPPER_OPTIONS = {
    "filein",
    "fileout",
    "informat",
    "outformat",
    "split",
    "join",
    "scheduler",
    "client",
    "workers",
    "threads",
    "tasks",
    "tempdir",
    "keep",
    "verb",
    "verbose",
    "data",
    "headers",
    "hdrcopy",
    "hfile",
    "bfile",
    "chunk",
    "name",
}


def _yes(value: Any) -> bool:
    return str(value).lower() in ("1", "y", "yes", "true", "on")


def balanced_partitions(size: int, jobs: int) -> List[Partition]:
    """Return sf_split-compatible balanced, ordered partitions."""
    if size < 1:
        raise DaskRsfError("split axis is empty")
    if jobs < 1:
        raise DaskRsfError("tasks/workers must be positive")
    jobs = min(size, jobs)
    width = 1 + size // jobs if jobs < size else 1
    big_jobs = size - jobs * (width - 1)
    result: List[Partition] = []
    for job in range(jobs):
        if job < big_jobs:
            count = width
            start = job * width
        else:
            count = width - 1
            start = big_jobs * width + (job - big_jobs) * count
        result.append(Partition(job, start, count))
    return result


def parse_cli(argv: Sequence[str]) -> ParsedCommand:
    """Parse wrapper options and retain one quoted per-chunk pipeline."""
    args = list(argv)
    if "--" in args:
        marker = args.index("--")
        wrapper, command_args = args[:marker], args[marker + 1 :]
    else:
        marker = next(
            (
                i
                for i, arg in enumerate(args)
                if "=" not in arg and not arg.startswith("_")
            ),
            len(args),
        )
        wrapper, command_args = args[:marker], args[marker:]

    options: Dict[str, str] = {}
    aux_inputs: Dict[str, str] = {}
    aux_outputs: Dict[str, str] = {}
    for arg in wrapper:
        if "=" not in arg:
            raise DaskRsfError("expected key=value before --: %s" % arg)
        key, value = arg.split("=", 1)
        if key.startswith("__"):
            aux_outputs[key[2:]] = value
        elif key.startswith("_"):
            aux_inputs[key[1:]] = value
        elif key in ("--input", "in"):
            options["filein"] = value
        elif key in ("--output", "out"):
            options["fileout"] = value
        elif key in _WRAPPER_OPTIONS:
            options[key] = value
        else:
            raise DaskRsfError(
                "unknown sfdask option %s (wrapped-command options belong after --)"
                % key
            )

    if not command_args:
        raise DaskRsfError("need a Madagascar command after --")
    if len(command_args) == 1:
        command = command_args[0]
    else:
        command = shlex.join(command_args)
    if not command.strip():
        raise DaskRsfError("empty command pipeline")
    return ParsedCommand(options, aux_inputs, aux_outputs, command)


def _rsf_history(path: str) -> Mapping[str, str]:
    with open(path, "rb") as stream:
        raw = stream.read(1024 * 1024)
    text = raw.split(b"\f", 1)[0].decode("latin-1", "replace")
    values: Dict[str, str] = {}
    pattern = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)")
    for match in pattern.finditer(text):
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def rsf_dimensions(path: str) -> List[int]:
    history = _rsf_history(path)
    dimensions: List[int] = []
    axis = 1
    while "n%d" % axis in history:
        dimensions.append(int(history["n%d" % axis]))
        axis += 1
    if not dimensions:
        raise DaskRsfError("could not read RSF dimensions from %s" % path)
    return dimensions


def _run_checked(
    command: Sequence[str],
    *,
    stdin_path: Optional[str] = None,
    stdout_path: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    stdin = open(stdin_path, "rb") if stdin_path else None
    stdout = open(stdout_path, "wb") if stdout_path else subprocess.PIPE
    try:
        proc = subprocess.run(
            list(command),
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            env=dict(env) if env is not None else None,
            check=False,
        )
    finally:
        if stdin:
            stdin.close()
        if stdout_path and stdout:
            stdout.close()
    if proc.returncode:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise DaskRsfError("%s failed: %s" % (shlex.join(command), detail))


def _inject_first_command(command: str, assignments: Iterable[str]) -> str:
    additions = " ".join(assignments)
    if not additions:
        return command
    first, separator, rest = command.partition("|")
    updated = "%s %s" % (first.rstrip(), additions)
    return updated if not separator else "%s |%s" % (updated, rest)


def _execute_partition(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Dask worker entry point.  All values are deliberately serializable."""
    axis = int(spec["axis"])
    start = int(spec["start"])
    count = int(spec["count"])
    aux_assignments: List[str] = []
    worker_env = os.environ.copy()
    worker_env["DATAPATH"] = spec["datapath"] + os.sep

    for name, source, target in spec["aux_inputs"]:
        _run_checked(
            ["sfwindow", "f%d=%d" % (axis, start), "n%d=%d" % (axis, count)],
            stdin_path=source,
            stdout_path=target,
            env=worker_env,
        )
        aux_assignments.append("%s=%s" % (name, shlex.quote(target)))

    for name, target in spec["aux_outputs"]:
        aux_assignments.append("%s=%s" % (name, shlex.quote(target)))

    command = _inject_first_command(spec["command"], aux_assignments)
    if spec["informat"] == "mdio":
        producer = "sfmdioread read=d mdio=%s f%d=%d n%d=%d" % (
            shlex.quote(spec["input"]),
            axis,
            start,
            axis,
            count,
        )
    else:
        producer = "sfwindow f%d=%d n%d=%d < %s" % (
            axis,
            start,
            axis,
            count,
            shlex.quote(spec["input"]),
        )

    script = "%s | %s > %s" % (
        producer,
        command,
        shlex.quote(spec["output"]),
    )
    proc = subprocess.run(
        ["bash", "-o", "pipefail", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=worker_env,
        check=False,
    )
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode:
        raise DaskRsfError(
            "partition %d failed (%s):\n%s"
            % (int(spec["index"]), script, stderr.strip())
        )
    return {"index": int(spec["index"]), "stderr": stderr}


def _patch_mdio(spec: Mapping[str, Any]) -> int:
    command = [
        "sfmdiowrite",
        "mode=update",
        "mdio=%s" % spec["mdio"],
        "f%d=%d" % (int(spec["axis"]), int(spec["start"])),
    ]
    if spec.get("tfile"):
        command.append("tfile=%s" % spec["tfile"])
    _run_checked(command, stdin_path=spec["input"])
    return int(spec["index"])


def _merge_rsf(
    inputs: Sequence[str],
    output: str,
    join_axis: int,
    datapath: Optional[str] = None,
) -> None:
    if not inputs:
        raise DaskRsfError("no partition outputs to merge")
    if join_axis == 0:
        command = ["sfadd"] + list(inputs[1:])
    else:
        command = ["sfcat", "axis=%d" % join_axis] + list(inputs[1:])
    env = os.environ.copy()
    if datapath:
        env["DATAPATH"] = datapath + os.sep
    _run_checked(command, stdin_path=inputs[0], stdout_path=output, env=env)


def _stream_rsf(path: str) -> None:
    with open(path, "rb") as source:
        proc = subprocess.run(
            ["sfcat"],
            stdin=source,
            stdout=sys.stdout.buffer,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode:
        raise DaskRsfError(proc.stderr.decode("utf-8", "replace"))


def _client_task_count(client: Any, requested: Optional[int]) -> int:
    if requested is not None:
        return requested
    info = client.scheduler_info()
    threads = sum(int(worker.get("nthreads", 1)) for worker in info["workers"].values())
    return max(1, threads)


def _mdio_header(path: str, target: str, data: Optional[str] = None) -> None:
    command = ["sfmdioread", "read=d", "mdio=%s" % path, "--dryrun=y"]
    if data:
        command.append("data=%s" % data)
    _run_checked(command, stdout_path=target)


def _mdio_common_args(options: Mapping[str, str]) -> List[str]:
    args: List[str] = []
    for key in ("data", "headers", "hdrcopy", "hfile", "bfile", "chunk", "name"):
        if key in options:
            args.append("%s=%s" % (key, options[key]))
    return args


def run(
    command: str,
    *,
    filein: Optional[str] = None,
    fileout: Optional[str] = None,
    informat: str = "rsf",
    outformat: str = "rsf",
    split: Optional[int] = None,
    join: Optional[int] = None,
    client: Any = None,
    scheduler: Optional[str] = None,
    workers: Optional[int] = None,
    threads: Optional[int] = None,
    tasks: Optional[int] = None,
    tempdir: Optional[str] = None,
    keep: bool = False,
    verbose: bool = False,
    aux_inputs: Optional[Mapping[str, str]] = None,
    aux_outputs: Optional[Mapping[str, str]] = None,
    mdio_options: Optional[Mapping[str, str]] = None,
) -> str:
    """Run *command* over partitions; return the output path."""
    if informat not in ("rsf", "mdio") or outformat not in ("rsf", "mdio"):
        raise DaskRsfError("informat/outformat must be rsf or mdio")
    if client is not None and scheduler:
        raise DaskRsfError("pass either a Client or scheduler=, not both")

    try:
        from distributed import Client
    except ImportError as exc:
        raise DaskRsfError(
            "Dask Distributed is required; run `scons setup` in user/mark"
        ) from exc

    base = tempdir or os.environ.get("TMPDATAPATH") or os.environ.get("DATAPATH")
    if base:
        base = os.path.abspath(os.path.expanduser(base))
        os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix="sfdask-", dir=base)
    own_client = client is None
    dask_client = client
    output_result = fileout
    try:
        if dask_client is None:
            kwargs: Dict[str, Any] = {}
            if workers is not None:
                kwargs["n_workers"] = workers
            if threads is not None:
                kwargs["threads_per_worker"] = threads
            dask_client = Client(scheduler, **kwargs) if scheduler else Client(**kwargs)

        if informat == "rsf":
            if filein:
                input_path = os.path.abspath(filein)
            else:
                input_path = os.path.join(work, "stdin.rsf")
                with open(input_path, "wb") as target:
                    shutil.copyfileobj(sys.stdin.buffer, target)
        else:
            if not filein:
                raise DaskRsfError("MDIO input requires filein=")
            input_path = filein

        header_path = input_path
        if informat == "mdio":
            header_path = os.path.join(work, "mdio-input.rsf")
            _mdio_header(input_path, header_path, (mdio_options or {}).get("data"))
        dimensions = rsf_dimensions(header_path)
        split_axis = split or len(dimensions)
        if split_axis < 1 or split_axis > len(dimensions):
            raise DaskRsfError("split=%d exceeds input rank %d" % (split_axis, len(dimensions)))
        join_axis = split_axis if join is None else join
        if join_axis < 0:
            raise DaskRsfError("join must be zero or a positive axis")

        partitions = balanced_partitions(
            dimensions[split_axis - 1], _client_task_count(dask_client, tasks)
        )
        if verbose:
            print(
                "sfdask: %d partitions along axis %d" % (len(partitions), split_axis),
                file=sys.stderr,
            )

        aux_in = dict(aux_inputs or {})
        aux_out = dict(aux_outputs or {})
        specs: List[Dict[str, Any]] = []
        primary_outputs: List[str] = []
        auxiliary_chunks: Dict[str, List[str]] = {name: [] for name in aux_out}
        for part in partitions:
            primary = os.path.join(work, "part-%06d.rsf" % part.index)
            primary_outputs.append(primary)
            split_inputs = []
            for name, source in aux_in.items():
                target = os.path.join(work, "%s-in-%06d.rsf" % (name, part.index))
                split_inputs.append((name, os.path.abspath(source), target))
            split_outputs = []
            for name in aux_out:
                target = os.path.join(work, "%s-out-%06d.rsf" % (name, part.index))
                auxiliary_chunks[name].append(target)
                split_outputs.append((name, target))
            specs.append(
                {
                    "index": part.index,
                    "axis": split_axis,
                    "start": part.start,
                    "count": part.count,
                    "informat": informat,
                    "input": input_path,
                    "output": primary,
                    "command": command,
                    "datapath": work,
                    "aux_inputs": split_inputs,
                    "aux_outputs": split_outputs,
                }
            )

        results = dask_client.gather(
            [dask_client.submit(_execute_partition, spec, pure=False) for spec in specs]
        )
        if verbose:
            for result in sorted(results, key=lambda item: item["index"]):
                if result["stderr"]:
                    sys.stderr.write(result["stderr"])

        for name, target in aux_out.items():
            _merge_rsf(auxiliary_chunks[name], os.path.abspath(target), join_axis)

        if outformat == "rsf":
            merged = os.path.abspath(fileout) if fileout else os.path.join(work, "stdout.rsf")
            _merge_rsf(primary_outputs, merged, join_axis)
            if not fileout:
                _stream_rsf(merged)
            output_result = fileout or "-"
        elif join_axis == 0:
            reduced = os.path.join(work, "reduced.rsf")
            _merge_rsf(primary_outputs, reduced, 0, datapath=work)
            if not fileout:
                raise DaskRsfError("MDIO output requires fileout=")
            command_args = ["sfmdiowrite", "mdio=%s" % fileout]
            command_args.extend(_mdio_common_args(mdio_options or {}))
            _run_checked(command_args, stdin_path=reduced)
            output_result = fileout
        else:
            if not fileout:
                raise DaskRsfError("MDIO output requires fileout=")
            template = os.path.join(work, "output-template.rsf")
            _merge_rsf(primary_outputs, template, join_axis, datapath=work)
            init = [
                "sfmdiowrite",
                "mode=init",
                "mdio=%s" % fileout,
                "chunk%d=1" % join_axis,
            ]
            init.extend(_mdio_common_args(mdio_options or {}))
            if "tfile" in aux_out:
                init.append("tfile=%s" % os.path.abspath(aux_out["tfile"]))
            _run_checked(init, stdin_path=template)

            offset = 0
            patches = []
            for spec, primary in zip(specs, primary_outputs):
                out_dims = rsf_dimensions(primary)
                if join_axis > len(out_dims):
                    raise DaskRsfError("join axis exceeds command output rank")
                patches.append(
                    {
                        "index": spec["index"],
                        "axis": join_axis,
                        "start": offset,
                        "input": primary,
                        "mdio": fileout,
                        "tfile": (
                            auxiliary_chunks["tfile"][spec["index"]]
                            if "tfile" in auxiliary_chunks
                            else None
                        ),
                    }
                )
                offset += out_dims[join_axis - 1]
            dask_client.gather(
                [dask_client.submit(_patch_mdio, patch, pure=False) for patch in patches]
            )
            output_result = fileout
        return output_result or "-"
    finally:
        if own_client and dask_client is not None:
            dask_client.close()
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        parsed = parse_cli(sys.argv[1:] if argv is None else argv)
        options = parsed.options
        run(
            parsed.command,
            filein=options.get("filein"),
            fileout=options.get("fileout"),
            informat=options.get("informat", "rsf").lower(),
            outformat=options.get("outformat", "rsf").lower(),
            split=int(options["split"]) if "split" in options else None,
            join=int(options["join"]) if "join" in options else None,
            scheduler=options.get("scheduler") or options.get("client"),
            workers=int(options["workers"]) if "workers" in options else None,
            threads=int(options["threads"]) if "threads" in options else None,
            tasks=int(options["tasks"]) if "tasks" in options else None,
            tempdir=options.get("tempdir"),
            keep=_yes(options.get("keep", "n")),
            verbose=_yes(options.get("verb", options.get("verbose", "n"))),
            aux_inputs=parsed.aux_inputs,
            aux_outputs=parsed.aux_outputs,
            mdio_options={
                key: value
                for key, value in options.items()
                if key in ("data", "headers", "hdrcopy", "hfile", "bfile", "chunk", "name")
            },
        )
        return 0
    except (DaskRsfError, ValueError) as exc:
        print("sfdask: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
