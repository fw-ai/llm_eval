from typing import Iterator, List

from .common import Task, Prompt, analyze_code, load_cacheable_df


class HumanevalTask(Task):
    def __init__(self, ds_cache_dir: str):
        self.ds_cache_dir = ds_cache_dir

    def get_prompts(self) -> Iterator[Prompt]:
        df = load_cacheable_df(
            name="humaneval",
            ds_cache_dir=self.ds_cache_dir,
            hf_ds_name="openai_humaneval",
            hf_split="test",
        )
        for _, data in df.iterrows():
            yield Prompt(
                data["prompt"],
                labels={
                    "task_id": data["task_id"],
                    "prompt": data["prompt"],
                    "test": data["test"],
                    "entry_point": data["entry_point"],
                },
            )

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        ids = []
        test_cases = []
        log_probs = []
        for response in responses:
            ids.append(response["labels"]["task_id"])
            log_probs.append(response["choices"][0]["logprobs"]["token_logprobs"])
            test_cases.append(
                response["labels"]["prompt"]
                + "\n"
                + response["choices"][0]["text"]
                + "\n"
                + response["labels"]["test"]
                + "\n\ncheck("
                + response["labels"]["entry_point"]
                + ")\n"
            )
        analyze_code(ids, test_cases, log_probs, verbose=verbose)

    @property
    def max_tokens(self):
        return 1024

    @property
    def stop_words(self) -> List[str]:
        return ["\nclass", "\ndef"]
