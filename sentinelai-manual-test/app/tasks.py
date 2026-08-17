"""Background task helpers - each function is an intentional, separate
vulnerability class for benchmarking: unsafe deserialization, command
injection (two forms), insecure YAML loading, and arbitrary code execution.
"""
import os
import pickle
import subprocess

import yaml


def load_cached_result(blob: bytes):
    return pickle.loads(blob)


def ping_host(host: str) -> None:
    os.system("ping -c 1 " + host)


def run_backup(cmd: str) -> None:
    subprocess.call(cmd, shell=True)


def load_task_config(text: str):
    return yaml.load(text)


def evaluate_expression(expr: str):
    return eval(expr)
