import pathlib
import os
from typing import Dict, Type

from .common import Prompt, Task
from .generic import GenericTask
from .mmlu import MmluTask
from .replay import ReplayTask
from .humaneval import HumanevalTask


def get_task(
    name: str,
    ds_cache_dir: str | None = None,
    prompt_style: str = "base",
    task_args: Dict = {},
) -> Task:
    if ds_cache_dir is None:
        ds_cache_dir = os.path.join(pathlib.Path.home(), ".cache/fireworks/datasets")

    cls = get_task_class(name)
    if cls == MmluTask:
        return MmluTask(ds_cache_dir, prompt_style)
    elif cls == HumanevalTask:
        return HumanevalTask(ds_cache_dir)
    elif cls == GenericTask:
        return GenericTask(task_args["generic_ds_name"], ds_cache_dir, prompt_style)
    elif cls == ReplayTask:
        return ReplayTask(task_args["replay_file"])
    else:
        raise ValueError(f"Unsupported task {name}")


def get_task_class(name: str) -> Type[Task]:
    if name == "mmlu":
        return MmluTask
    elif name == "humaneval":
        return HumanevalTask
    elif name == "generic":
        return GenericTask
    elif name == "replay":
        return ReplayTask
    else:
        raise ValueError(f"Unsupported task {name}")
