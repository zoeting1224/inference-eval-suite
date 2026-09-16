"""tmux lifecycle for long-running inferbench commands."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def session_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("task name must contain only letters, digits, dot, dash or underscore")
    return "inferbench-" + name


def exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", "=" + session_name(name)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def start(root: Path, name: str, command: list[str]) -> None:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("task start requires an inferbench subcommand after --")
    if command[0] == "task":
        raise ValueError("a background task cannot recursively start another task")
    if exists(name):
        raise ValueError(f"task already exists: {session_name(name)}")
    full = [sys.executable, str(root / "inferbench.py"), *command]
    pane = subprocess.check_output(
        ["tmux", "new-session", "-d", "-P", "-F", "#{pane_id}",
         "-s", session_name(name), "-c", str(root)],
        text=True,
    ).strip()
    subprocess.run(["tmux", "set-option", "-w", "-t", pane, "remain-on-exit", "on"], check=True)
    subprocess.run(["tmux", "send-keys", "-t", pane, "-l", shlex.join(full)], check=True)
    subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], check=True)
    print(f"Started: {session_name(name)}")
    print(f"Attach:  tmux attach -t {session_name(name)}")


def status(name: str | None = None) -> None:
    if name:
        subprocess.run(
            ["tmux", "list-panes", "-t", "=" + session_name(name),
             "-F", "session=#{session_name} pane=#{pane_id} dead=#{pane_dead} pid=#{pane_pid} command=#{pane_current_command}"],
            check=True,
        )
        return
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("inferbench-"):
            print(line)


def logs(name: str, lines: int = 200) -> None:
    subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", "=" + session_name(name), "-S", f"-{lines}"],
        check=True,
    )


def stop(name: str) -> None:
    if not exists(name):
        raise ValueError(f"task not found: {session_name(name)}")
    subprocess.run(["tmux", "send-keys", "-t", "=" + session_name(name), "C-c"], check=True)
    print(f"Interrupt sent: {session_name(name)}")
    print("Managed searches will clean up only the container they own.")


def attach(name: str) -> None:
    os.execvp("tmux", ["tmux", "attach", "-t", session_name(name)])
