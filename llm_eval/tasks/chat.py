import json
import os
from typing import Iterator
import transformers

from .common import Task, Prompt, analyze_generic


class ChatTask(Task):
    def __init__(self, ds_file: str, tokenizer_dir: str | None):
        self.ds_file = ds_file
        if tokenizer_dir is not None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = None

    def get_prompts(self) -> Iterator[Prompt]:
        with open(self.ds_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if self.tokenizer is not None:
                    tokens = self.tokenizer.apply_chat_template(
                        data["messages"], tokenize=True
                    )
                    yield Prompt(tokens)
                else:
                    yield Prompt("", messages=data["messages"])

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        analyze_generic(responses, verbose)

    @property
    def echo(self) -> bool:
        return True
