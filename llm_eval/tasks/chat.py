import json
import os
from typing import Iterator
import transformers

from .common import Task, Prompt, analyze_generic


class ChatTask(Task):
    def __init__(self, ds_file: str, tokenizer_dir: str):
        self.ds_file = ds_file
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)

    def get_prompts(self) -> Iterator[Prompt]:
        with open(self.ds_file, "r") as f:
            for line in f:
                data = json.loads(line)
                tokens = self.tokenizer.apply_chat_template(
                    data["messages"], tokenize=True
                )
                yield Prompt(tokens)

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        analyze_generic(responses, verbose)

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True
