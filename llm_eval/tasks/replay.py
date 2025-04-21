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

                assert "raw_output" in choice
                if "completion_logprobs" in choice["raw_output"] and choice["raw_output"]["completion_logprobs"] is not None:
                    prompt_tokens = choice["raw_output"]["prompt_token_ids"]
                    all_tokens = [
                        x["token_id"]
                        for x in choice["raw_output"]["completion_logprobs"]["content"]
                    ]
                    yield Prompt(
                        text=prompt_tokens,
                        forced_generation=all_tokens[len(prompt_tokens) :],
                    )
                else:
                    prompt = choice["raw_output"]["prompt_fragments"][0]
                    completion = choice["raw_output"]["completion"][len(prompt) :]
                    yield Prompt(text=prompt, forced_generation=completion)

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True
