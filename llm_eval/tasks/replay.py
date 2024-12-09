import json
from typing import Iterator

from .common import Task, Prompt


class ReplayTask(Task):
    def __init__(self, replay_file: str):
        self.replay_file = replay_file

    def get_prompts(self) -> Iterator[Prompt]:
        with open(self.replay_file, "r") as f:
            for line in f:
                resp = json.loads(line)
                choice = resp["choices"][0]
                n_prompt = resp["usage"]["prompt_tokens"]

                tokens = choice["logprobs"]["tokens"]

                prompt_tokens = choice["raw_output"]["prompt_token_ids"]
                all_tokens = [
                    x["token_id"]
                    for x in choice["raw_output"]["completion_logprobs"]["content"]
                ]
                yield Prompt(
                    text=prompt_tokens,
                    forced_generation=all_tokens[len(prompt_tokens) :],
                )

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True
