from typing import IO, List
import argparse
from dataclasses import dataclass
import json
import asyncio
import logging
import os
import time
import functools

import tqdm
import httpx
import transformers
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp

from .tasks import get_task, Task, Prompt


logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class GeneratorArgs:
    mode: str = ""

    # HF arguments.
    hf_dir = ""

    # URL arguments.
    host: str = "http://localhost:80"
    model_name: str | None = None
    batch_size: int = 64
    auth_token: str | None = None

    # Task common arguments.
    task_name: str = ""
    out_file: str = "responses.jsonl"
    limit: int | None = None  # no limit
    ds_cache_dir: str | None = None
    prompt_style: str = "base"

    # LLM parameters settable by a task
    # and overridable from command-line.
    max_tokens: int = None
    logprobs: int | None = None
    echo: bool | None = None
    stop_words: List[str] | None = None

    # Task-specific.
    generic_ds_name: str | None = None
    replay_file: str | None = None


def _get_task(args: GeneratorArgs) -> Task:
    task_args = {
        "generic_ds_name": args.generic_ds_name,
        "replay_file": args.replay_file,
    }
    return get_task(
        name=args.task_name,
        ds_cache_dir=args.ds_cache_dir,
        prompt_style=args.prompt_style,
        task_args=task_args,
    )


def _get_prompts(task: Task, args: GeneratorArgs) -> List[Prompt]:
    prompts = list(task.get_prompts())
    if args.limit is not None:
        prompts = prompts[: args.limit]
    return prompts


async def _send_prompts_url(args: GeneratorArgs, f: IO):
    task = _get_task(args)
    prompts = _get_prompts(task, args)

    client = httpx.AsyncClient(timeout=120)
    request_aws = set()
    curr_in_id = 0
    curr_out_id = [0]

    headers = {}
    if args.auth_token is not None:
        headers["Authorization"] = f"Bearer {args.auth_token}"

    async def _post(id, rdata):
        res = await client.post(
            args.host + "/v1/completions", json=rdata, headers=headers
        )
        return id, res.json()

    responses = []
    all_responses = []

    def _dump_responses():
        responses.sort(key=lambda x: x[0])
        while responses and responses[0][0] == curr_out_id[0]:
            resp_id, response = responses.pop(0)
            response["id"] = str(resp_id)
            if f is not None:
                f.write(json.dumps(response))
                f.write("\n")
                f.flush()
            curr_out_id[0] += 1
            all_responses.append(response)

    def _set_param(rdata, rname: str, pname: str | None = None):
        if pname is None:
            pname = rname
        val = getattr(args, pname, None)
        if val is None:
            val = getattr(task, pname, None)
        if val is not None:
            rdata[rname] = val

    prompts = list(prompts)
    pbar = tqdm.trange(len(prompts))
    prompts_iter = iter(prompts)
    pending_prompts = {}
    while True:
        while len(request_aws) < args.batch_size:
            try:
                prompt = next(prompts_iter)
                pbar.update()
            except StopIteration:
                break

            rdata = {
                "prompt": prompt.text,
                "stream": False,
                "temperature": 0,
                "raw_output": True,
            }

            _set_param(rdata, "logprobs")
            _set_param(rdata, "max_tokens")
            _set_param(rdata, "echo")
            _set_param(rdata, "stop", "stop_words")

            if prompt.forced_generation:
                rdata["forced_generation"] = prompt.forced_generation

            if args.model_name:
                rdata["model"] = args.model_name

            request_aws.add(_post(curr_in_id, rdata))
            pending_prompts[curr_in_id] = prompt
            curr_in_id += 1

        if len(pending_prompts) == 0 and len(request_aws) == 0:
            break

        done_aws, request_aws = await asyncio.wait(
            request_aws, return_when=asyncio.FIRST_COMPLETED
        )
        for response_aws in done_aws:
            id, response = await response_aws
            prompt = pending_prompts.pop(id)
            response["labels"] = prompt.labels
            responses.append((id, response))
        _dump_responses()

    _dump_responses()

    task.analyze(all_responses)


class HFStreamingDetokenizer:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        prompt: List[int],
        buffer_size: int = 4,
    ):
        super().__init__()
        assert isinstance(prompt, list)
        self._tokenizer = tokenizer
        self._buffer_size = buffer_size
        self._replacement = chr(0xFFFD)
        self._buffer = prompt[-buffer_size - 1 :]
        self._surrogates = 0
        self._pos = 0

    def decode(self, token: int) -> str:
        """Decode token to string while handling surrogates and whitespace."""

        assert isinstance(token, int), type(token)

        # <unk>, <pad> and other special tokens will be decoded into ''.
        text = self.decode_skip_specials([token])

        # Handle replacement characters caused by multi-byte-pair-encoding or
        # Unicode surrogates or multi-code-point graphemes like emojis.
        if self._replacement in text:
            n = -self._surrogates if self._surrogates > 0 else len(self._buffer)
            tokens = self._buffer[n:] + [token]
            text = self.decode_skip_specials(tokens)

            # Check whether the last grapheme was successfully decoded.
            if text and text[-1] != self._replacement:
                text = text.replace(self._replacement, "")
                self._surrogates = 0
            else:
                text = ""
                self._surrogates += 1
        else:
            self._surrogates = 0

        # Handle whitespace between tokens.
        prefix = self.decode_skip_specials(self._buffer)
        tokens = self._buffer + [token]
        whole = self.decode_skip_specials(tokens)
        # Add extra space between non-space tokens.
        if prefix + " " + text == whole:
            text = " " + text

        # Update buffer and offsets.
        self._buffer = self._buffer[-self._buffer_size :] + [token]
        self._pos += len(text)

        return text

    @property
    def pos(self) -> int:
        return self._pos

    def decode_skip_specials(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


@torch.inference_mode()
def _send_prompts_hf(
    args: GeneratorArgs,
    hf_model: transformers.PreTrainedModel,
    hf_tokenizer: transformers.PreTrainedTokenizer,
    f: IO,
):
    assert args.stop_words is None
    assert not args.logprobs

    task = _get_task(args)
    prompts = _get_prompts(task, args)

    # Deduce device from model.
    param = hf_model.parameters().__next__()
    device = param.device

    def _get_param(pname: str, default):
        val = getattr(args, pname, None)
        if val is None:
            val = getattr(task, pname, None)
        return val if val is not None else default

    max_tokens = _get_param("max_tokens", 16)
    echo = _get_param("echo", False)

    responses = []
    for prompt_id, prompt in enumerate(tqdm.tqdm(prompts)):
        # NB: tokenizer settings about BOS and EOS here are tricky. Default HF
        # pipeline for text generation skips those tokens regardless of model's
        # tokenizer settings:
        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/pipelines/text_generation.py#L204
        # However, many models expect BOS token to be present in the input. It's
        # the case for Llama in particular:
        # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L192.
        # Hence we follow HF's model settings here and do not pass
        # "skip_special_tokens=True" to tokenizer.decode() below.
        inputs = hf_tokenizer(prompt.text, padding=False, return_tensors="pt")

        print("Feeding prompt of", inputs["input_ids"].numel(), "tokens")
        input_tokens = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        detokenized_prompt = hf_tokenizer.decode(
            inputs["input_ids"][0].tolist(),
            skip_special_tokens=False,
        )

        # NB: the prompt can be shorter because of the truncation above
        assert (
            detokenized_prompt == prompt.text[: len(detokenized_prompt)]
        ), "HF tokenizer didn't round-trip cleanly"

        # Run model on prompt once if echo is enabled,
        # HF generate() method doesn't return prompt logits.
        logprobs = []
        if echo:
            inputs = hf_model.prepare_inputs_for_generation(
                input_ids=input_tokens[:, :-1], attention_mask=attn_mask[:, :-1]
            )
            prefill_output = hf_model(**inputs, return_dict=True)
            prefill_logits = prefill_output.logits[0]

            logprobs.append(torch.zeros_like(prefill_logits[0]))
            for l in prefill_logits:
                prob = torch.nn.functional.softmax(l, dim=-1)
                logprobs.append(torch.log(prob + 1e-7))

        end_id = hf_tokenizer.eos_token_id or hf_tokenizer.im_end_id

        # Run model auto-regressively.
        incr_output = hf_model.generate(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            temperature=0,
            eos_token_id=end_id,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            do_sample=False,
            output_scores=True,
        )

        gen_tokens = incr_output.sequences[0]
        if not echo:
            gen_tokens = gen_tokens[input_tokens.numel() :]

        incr_logits = incr_output.scores
        for i in range(len(incr_logits)):
            prob = torch.nn.functional.softmax(incr_logits[i][0], dim=-1)
            logprobs.append(torch.log(prob + 1e-7))

        token_logprobs = []
        str_tokens = []
        text_offset = []
        stream_detokenizer = HFStreamingDetokenizer(
            hf_tokenizer, input_tokens.view(-1).tolist()
        )
        for token, logprob in zip(gen_tokens.tolist(), logprobs):
            token_logprobs.append(logprob[token].item())

            text_offset.append(stream_detokenizer.pos)
            str_tokens.append(stream_detokenizer.decode(token))

        text = hf_tokenizer.decode(gen_tokens, skip_special_tokens=False)

        if echo:
            assert (
                text[: len(detokenized_prompt)] == detokenized_prompt
            ), "HF tokenizer didn't round-trip cleanly"

        finish_reason = "stop" if gen_tokens[-1] == end_id else "length"

        completion_tokens = (
            (gen_tokens.numel() - input_tokens.numel()) if echo else gen_tokens.numel()
        )
        total_tokens = (
            gen_tokens.numel() if echo else (input_tokens.numel() + gen_tokens.numel())
        )
        response = {
            "id": f"{prompt_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": type(hf_model).__name__,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "tokens": str_tokens,
                        "token_logprobs": token_logprobs,
                        "top_logprobs": None,
                        "text_offset": text_offset,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens.numel(),
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        response["labels"] = prompt.labels
        responses.append(response)
        if f is not None:
            f.write(json.dumps(response))
            f.write("\n")

    task.analyze(responses)


def _create_hf_model(hf_dir: str) -> transformers.PreTrainedModel:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")

    hf_model: transformers.PreTrainedModel = (
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_dir,
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    )
    hf_model.eval()

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(device)
        # TODO: add layers here for large models,
        # which don't fit on a single GPU.
        wrap_policy = functools.partial(
            fsdp.wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformers.models.llama.modeling_llama.LlamaDecoderLayer,
                transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer,
            },
        )
        hf_model = fsdp.FullyShardedDataParallel(
            hf_model, device_id=device, auto_wrap_policy=wrap_policy
        )
    return hf_model.to(device)


def main():
    args = GeneratorArgs()
    parser = argparse.ArgumentParser(prog="generator")
    parser.add_argument(
        "mode",
        type=str,
        help="Mode to run: url (remote LLM) or hf (local Huggingface LLM)",
        choices=["url", "hf"],
    )

    # URL mode.
    url_group = parser.add_argument_group("url")
    url_group.add_argument("-H", "--host", default=args.host, type=str, help="Base URL")
    url_group.add_argument(
        "-n",
        "--model-name",
        type=str,
        default=args.model_name,
    )
    url_group.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to use",
        default=args.batch_size,
    )
    url_group.add_argument(
        "-t",
        "--auth-token",
        type=str,
        help="Authentication token",
        default=args.auth_token,
    )

    # HF mode.
    hf_group = parser.add_argument_group("hf")
    hf_group.add_argument("--hf-dir", type=str)

    def _add_task(parsers, name: str):
        parser = parsers.add_parser(name)
        # Add task common args.
        parser.add_argument(
            "-o",
            "--out-file",
            type=str,
            help="Path where to store responses",
            default=args.out_file,
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="Max tokens",
            default=args.max_tokens,
        )
        parser.add_argument(
            "--logprobs",
            type=int,
            help="Top log probs to retrieve",
        )
        parser.add_argument(
            "--echo",
            help="Whether to pass 'echo=true' to text completion API",
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "-l",
            "--limit",
            type=int,
            help="Max prompts",
            default=args.limit,
        )
        parser.add_argument("--stop-words", nargs="+", type=str)
        parser.add_argument("-p", "--prompt-style", type=str, default=args.prompt_style)
        parser.add_argument(
            "--ds-cache-dir",
            type=str,
            help="Override default cache dir",
            default=args.auth_token,
        )
        return parser

    # Create task parsers.
    task_parsers = parser.add_subparsers(dest="task_name", required=True)
    generic_parser = _add_task(task_parsers, "generic")
    generic_parser.add_argument(
        "-d",
        "--dataset",
        dest="generic_ds_name",
        required=True,
        type=str,
        help="Dataset name",
    )
    mmlu_parser = _add_task(task_parsers, "mmlu")
    humaneval = _add_task(task_parsers, "humaneval")
    replay_parser = _add_task(task_parsers, "replay")
    replay_parser.add_argument(
        "-f",
        "--replay-file",
        required=True,
        type=str,
        help="Responses file to replay",
    )

    parser.parse_args(namespace=args)

    with open(args.out_file, "w+") as f:
        if args.mode == "url":
            requests = _send_prompts_url(args, f)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(requests)
            loop.close()
        elif args.mode == "hf":
            assert args.hf_dir is not None, "--hf-dir is required for 'hf' mode"
            hf_model = _create_hf_model(args.hf_dir)
            hf_tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_dir)
            _send_prompts_hf(args, hf_model, hf_tokenizer, f)
        else:
            raise ValueError(f"Invalid mode {args.mode}")


if __name__ == "__main__":
    main()
