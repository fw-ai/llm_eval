import os
from typing import Iterator, Dict, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tempfile

from jinja2 import Template
import numpy
import numpy as np
import tabulate
import datasets
import pandas as pd
import tqdm

from .execution import check_correctness


@dataclass
class Prompt:
    text: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    forced_generation: str | List[int] = field(default_factory=list)


def _get_template_dir(task_name: str) -> str:
    curr_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(curr_dir, "templates", task_name)


def load_template(task_name: str, prompt_style: str | None = None):
    if not prompt_style:
        return Template("{{ text }}")

    template_file = os.path.join(_get_template_dir(task_name), prompt_style + ".jinja")
    if not os.path.isfile(template_file):
        raise ValueError(
            f"Unsupported prompt style {prompt_style} for dataset {task_name}"
        )
    with open(template_file) as f:
        template_str = f.read()
        return Template(template_str, keep_trailing_newline=True)


def _perplexity(resp):
    choice = resp["choices"][0]
    if "token_logprobs" in choice["logprobs"]:
        logprobs = choice["logprobs"]["token_logprobs"]
    else:
        logprobs = []
        for lp_content in choice["logprobs"]["content"]:
            logprobs.append(lp_content["logprob"])
    return numpy.exp(-numpy.mean(logprobs))


def get_dataset_dir(ds_name: str, ds_cache_dir: str) -> str | None:
    code_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), f"datasets/{ds_name}"
    )
    if os.path.exists(code_dir):
        return code_dir

    cache_dir = os.path.join(ds_cache_dir, ds_name)
    if os.path.exists(cache_dir):
        return cache_dir

    raise ValueError(f"No dataset folder found {code_dir} or {cache_dir}")


class Task:
    def get_prompts(self) -> Iterator[Prompt]:
        raise NotImplementedError()

    @property
    def max_tokens(self) -> int:
        return None

    @property
    def logprobs(self) -> int:
        return 0

    @property
    def echo(self) -> bool:
        return False

    @property
    def stop_words(self) -> List[str] | None:
        return None

    @classmethod
    def analyze(cls, responses: Iterator, verbose: bool = False):
        pass

    @property
    def use_chat(self) -> bool:
        return False


def analyze_multichoice(responses: Iterator, verbose=False):
    table = [["Pos", "ID", "Perplexity", "Accuracy"]]

    total_perplex = 0
    total_acc = 0
    num_responses = 0
    for i, response in enumerate(responses):
        num_responses += 1
        if "choices" not in response:
            continue

        perplex = _perplexity(response)
        total_perplex += perplex

        choice = response["choices"][0]
        answer = "".join(choice["text"]).strip()
        splits = answer.split("\n")
        if len(splits) > 1:
            answer = splits[0]
        acc = response["labels"]["correct_answer"] == answer

        total_acc += acc

        table.append(
            [
                i,
                response["id"],
                perplex,
                acc,
            ]
        )

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                ["Perplexity", total_perplex / num_responses],
                ["Accuracy", total_acc / num_responses],
            ],
            tablefmt="plain",
        )
    )


def analyze_generic(responses: List, verbose: bool):
    table = [["Pos", "ID", "Perplexity", "Prompt Tokens", "Gen Tokens"]]

    total_perplex = 0
    num_responses = 0
    prompt_lens = []
    gen_lens = []
    for i, response in enumerate(responses):
        if "choices" not in response:
            continue
        perplex = _perplexity(response)
        total_perplex += perplex
        prompt_lens.append(response["usage"]["prompt_tokens"])
        gen_lens.append(response["usage"]["completion_tokens"])

        num_responses += 1

        table.append(
            [
                i,
                response["id"],
                perplex,
                prompt_lens[-1],
                gen_lens[-1],
            ]
        )

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                [
                    "Perplexity",
                    total_perplex / num_responses,
                ],
                [
                    "Avg Prompt Tokens",
                    sum(prompt_lens) / len(prompt_lens),
                ],
                [
                    "Avg Completion Tokens",
                    sum(gen_lens) / len(gen_lens),
                ],
            ],
            tablefmt="plain",
        )
    )


def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    See HumanEval paper: https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def analyze_code(
    ids: List[str],
    test_cases: List[str],
    logprobs: List[List[int]],
    num_workers=8,
    timeout=3.0,
    verbose=False,
):
    passed = [0]
    table = [["ID", "Passed"]]

    def _on_done(future, id, ppl):
        result = future.result()
        if result["passed"]:
            passed[0] += 1
        table.append([id, result["passed"], ppl])

    total_ppl = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for id, test_case, lp in zip(ids, test_cases, logprobs):
            ppl = numpy.exp(-numpy.mean(lp))
            total_ppl += ppl

            args = (test_case, timeout)
            future = executor.submit(check_correctness, *args)
            future.add_done_callback(partial(_on_done, id=id, ppl=ppl))

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                ["Perplexity", total_ppl / len(ids)],
                ["Pass@1", _pass_at_k(len(ids), passed[0], 1)],
            ],
            tablefmt="plain",
        )
    )


def load_cacheable_df(
    name: str,
    ds_cache_dir: str,
    hf_ds_name: str,
    hf_config: str | None = None,
    hf_split: str | None = None,
) -> pd.DataFrame:
    ds_folder = os.path.join(ds_cache_dir, name)
    cache_file = os.path.join(
        ds_folder,
        f"{hf_ds_name.replace('/', '_')}{ '-' + hf_config if hf_config else ''}{'-' + hf_split if hf_split else ''}.parquet",
    )
    if not os.path.exists(cache_file):
        ds = datasets.load_dataset(
            hf_ds_name, hf_config, split=hf_split, streaming=True, trust_remote_code=True
        )
        rows = []
        for row in tqdm.tqdm(ds):
            rows.append(row)
        df = pd.DataFrame(rows)
        os.makedirs(ds_folder, exist_ok=True)
        df.to_parquet(cache_file)

    return pd.read_parquet(cache_file)
