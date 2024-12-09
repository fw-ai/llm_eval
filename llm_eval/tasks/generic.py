import json
import os
from typing import Iterator

from .common import Task, Prompt, load_template, _get_template_dir, get_dataset_dir, analyze_generic

class GenericTask(Task):
    def __init__(self, ds_name: str, ds_cache_dir: str, prompt_style: str = "base"):
        self.ds_name = ds_name
        self.ds_cache_dir = ds_cache_dir
        self.prompt_style = prompt_style

    def get_prompts(self) -> Iterator[Prompt]:
        subtask = os.path.join("generic", self.ds_name)
        if os.path.exists(_get_template_dir(subtask)):
            template = load_template(subtask, self.prompt_style)
        else:
            template = load_template("generic", self.prompt_style)

        # Format prompts.
        ds_dir = get_dataset_dir(self.ds_name, self.ds_cache_dir)
        with open(os.path.join(ds_dir, "test.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                prompt = Prompt(template.render(**data))
                yield prompt

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        analyze_generic(responses, verbose)

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True
