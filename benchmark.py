#!/usr/bin/env python3

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


# psutil usually translates process-access failures to one of its own error
# classes, but macOS sandboxing can surface a bare OSError from sysctl instead.
# Treat either form as an unavailable sample, rather than letting monitoring
# terminate the benchmarked command.
PROCESS_ACCESS_ERRORS = (
    psutil.Error,
    OSError,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def get_tree(pid):
    try:
        root = psutil.Process(pid)
    except PROCESS_ACCESS_ERRORS:
        return []

    processes = [root]

    try:
        processes.extend(root.children(recursive=True))
    except PROCESS_ACCESS_ERRORS:
        pass

    running_processes = []

    for process in processes:
        try:
            if process.is_running():
                running_processes.append(process)
        except PROCESS_ACCESS_ERRORS:
            continue

    return running_processes


def get_cpu_time_and_memory(pid):

    total_cpu_time = 0.0
    total_rss = 0
    names = []

    for p in get_tree(pid):

        try:

            times = p.cpu_times()

            total_cpu_time += (
                times.user +
                times.system
            )

            total_rss += (
                p.memory_info().rss
            )

            names.append(
                f"{p.pid}:{p.name()}"
            )

        except PROCESS_ACCESS_ERRORS:
            pass

    return (
        total_cpu_time,
        total_rss,
        names
    )


def sample_gpu():

    if platform.system() != "Darwin":
        return {}

    try:

        result = subprocess.run(
            [
                "ioreg",
                "-r",
                "-c",
                "AGXAccelerator",
                "-w",
                "0"
            ],
            capture_output=True,
            text=True,
            timeout=2
        )

        text = result.stdout

    except Exception:
        return {}

    patterns = {

        "device_utilization_percent":
            r'"Device Utilization %"=([0-9.]+)',

        "renderer_utilization_percent":
            r'"Renderer Utilization %"=([0-9.]+)',

        "tiler_utilization_percent":
            r'"Tiler Utilization %"=([0-9.]+)',

    }

    stats = {}

    for name, pattern in patterns.items():

        match = re.search(
            pattern,
            text
        )

        if match:

            stats[name] = float(
                match.group(1)
            )

    return stats


def write_csv(rows, path):

    if not rows:
        return

    keys = []

    for row in rows:

        for key in row:

            if key not in keys:
                keys.append(key)

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(rows)


def average(values):

    if not values:
        return None

    return round(
        sum(values) / len(values),
        2
    )


def peak(values):

    if not values:
        return None

    return round(
        max(values),
        2
    )


def main():

    parser = argparse.ArgumentParser(
        description="STTL OCR benchmark"
    )

    parser.add_argument(
        "--name",
        required=True
    )

    parser.add_argument(
        "--pages",
        type=int,
        required=True
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    command = args.command

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error(
            "OCR command required after --"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    name = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        args.name
    )

    log_file = (
        output_dir /
        f"{name}.log"
    )

    samples_file = (
        output_dir /
        f"{name}_samples.csv"
    )

    summary_file = (
        output_dir /
        f"{name}_summary.json"
    )

    logical_cpus = (
        psutil.cpu_count(
            logical=True
        )
        or 1
    )

    print()
    print("=" * 70)
    print(
        f"STTL OCR BENCHMARK — "
        f"{args.name.upper()}"
    )
    print("=" * 70)
    print()

    print(
        "Logical CPU cores:",
        logical_cpus
    )

    print(
        "Command:",
        " ".join(command)
    )

    print()

    start = time.perf_counter()

    log = open(
        log_file,
        "w"
    )

    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT
    )

    samples = []

    while process.poll() is None:

        time.sleep(
            args.sample_interval
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        cpu_time, rss, names = (
            get_cpu_time_and_memory(
                process.pid
            )
        )

        memory = (
            psutil.virtual_memory()
        )

        gpu = sample_gpu()

        row = {

            "elapsed_seconds":
                round(elapsed, 3),

            "process_tree_cpu_seconds":
                round(cpu_time, 3),

            "process_tree_rss_bytes":
                rss,

            "system_memory_percent":
                memory.percent,

            "system_memory_used_bytes":
                memory.used,

            "system_memory_available_bytes":
                memory.available,

            "processes":
                "|".join(names),

        }

        row.update(gpu)

        samples.append(row)

    return_code = process.wait()

    # Capture final process CPU time.
    final_cpu_time, final_rss, final_names = (
        get_cpu_time_and_memory(
            process.pid
        )
    )

    log.close()

    runtime = (
        time.perf_counter()
        - start
    )

    # CPU time may disappear after process termination,
    # so use the largest observed value.
    cpu_times = [
        x["process_tree_cpu_seconds"]
        for x in samples
    ]

    cpu_seconds = max(
        cpu_times + [final_cpu_time]
    )

    cpu_utilization = (
        cpu_seconds /
        (runtime * logical_cpus)
        * 100
    )

    rss_values = [
        x["process_tree_rss_bytes"]
        for x in samples
    ]

    memory_values = [
        x["system_memory_percent"]
        for x in samples
    ]

    device_gpu = [
        x["device_utilization_percent"]
        for x in samples
        if "device_utilization_percent"
        in x
    ]

    renderer_gpu = [
        x["renderer_utilization_percent"]
        for x in samples
        if "renderer_utilization_percent"
        in x
    ]

    tiler_gpu = [
        x["tiler_utilization_percent"]
        for x in samples
        if "tiler_utilization_percent"
        in x
    ]

    summary = {

        "benchmark":
            args.name,

        "command":
            command,

        "pages":
            args.pages,

        "logical_cpu_cores":
            logical_cpus,

        "runtime_seconds":
            round(
                runtime,
                3
            ),

        "pages_per_second":
            round(
                args.pages /
                runtime,
                4
            ),

        "return_code":
            return_code,

        "success":
            return_code == 0,

        "cpu": {

            "total_cpu_seconds":
                round(
                    cpu_seconds,
                    3
                ),

            "average_utilization_percent":
                round(
                    cpu_utilization,
                    2
                ),

        },

        "process_memory": {

            "average_gb":
                round(
                    sum(rss_values) /
                    len(rss_values) /
                    (1024 ** 3),
                    3
                )
                if rss_values
                else None,

            "peak_gb":
                round(
                    max(rss_values) /
                    (1024 ** 3),
                    3
                )
                if rss_values
                else None,

        },

        "system_memory": {

            "average_percent":
                average(
                    memory_values
                ),

            "peak_percent":
                peak(
                    memory_values
                ),

        },

        "apple_gpu": {

            "device_average_percent":
                average(
                    device_gpu
                ),

            "device_peak_percent":
                peak(
                    device_gpu
                ),

            "renderer_average_percent":
                average(
                    renderer_gpu
                ),

            "renderer_peak_percent":
                peak(
                    renderer_gpu
                ),

            "tiler_average_percent":
                average(
                    tiler_gpu
                ),

            "tiler_peak_percent":
                peak(
                    tiler_gpu
                ),

            "measurement":
                "macOS ioreg AGXAccelerator; system-wide",

        },

        "sampling": {

            "interval_seconds":
                args.sample_interval,

            "sample_count":
                len(samples),

        },

    }

    write_csv(
        samples,
        samples_file
    )

    with open(
        summary_file,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()

    print(
        f"Runtime:             "
        f"{runtime:.2f} sec"
    )

    print(
        f"Pages/sec:           "
        f"{args.pages / runtime:.3f}"
    )

    print(
        f"CPU time:            "
        f"{cpu_seconds:.2f} sec"
    )

    print(
        f"Avg CPU utilization: "
        f"{cpu_utilization:.2f}%"
    )

    print(
        f"RAM average:         "
        f"{summary['process_memory']['average_gb']} GB"
    )

    print(
        f"RAM peak:            "
        f"{summary['process_memory']['peak_gb']} GB"
    )

    print(
        f"GPU device avg:      "
        f"{average(device_gpu)}%"
    )

    print(
        f"GPU device peak:     "
        f"{peak(device_gpu)}%"
    )

    print(
        f"GPU renderer avg:    "
        f"{average(renderer_gpu)}%"
    )

    print(
        f"GPU renderer peak:   "
        f"{peak(renderer_gpu)}%"
    )

    print(
        f"GPU tiler avg:       "
        f"{average(tiler_gpu)}%"
    )

    print(
        f"GPU tiler peak:      "
        f"{peak(tiler_gpu)}%"
    )

    print()
    print("Output files:")
    print(log_file)
    print(samples_file)
    print(summary_file)
    print()


if __name__ == "__main__":
    main()
