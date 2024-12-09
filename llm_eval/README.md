# LLM Evaluation Suite Usage

Evaluation is run by choosing one of two modes: url or hf and one of numerous tasks:
```
generator <mode> [mode args..] <task> [task args..]
```

## URL Mode
This mode allows to query remote LLM via OpenAI compatible 'completions' API.

```bash
python -m llm_eval.generator url generic --dataset fastchat
```

## Huggingface Mode
This mode creates and queries local Huggingface model.

```bash
python -m llm_eval.generator hf --hf-dir Meta-Llama-3-8B generic --dataset fastchat
```


## Simple Tasks

### MMLU
Runs [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) benchmark.

Running against a base model:

```bash
python -m llm_eval.generator url mmlu
```

LLaMA V3 instruct model:

```bash
python -m llm_eval.generator url mmlu --prompt-style llama-v3-instruct
```

### HumanEval
Runs [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374) benchmark.

```bash
python -m llm_eval.generator url humaneval
```

### Generic

Just issues prompts from a given dataset.
Example for a builtin 'fastchat' dataset:
```bash
python -m llm_eval.generator url generic -o fastchat.jsonl --dataset fastchat
```

Using custom dataset.
Assuming /my-datasets/my-numbers/test.jsonl file has newline-delimited json objects w/ "prompt" attribute, e.g.
```
{"prompt": "one two three"}
{"prompt": "1 2 3 4 5"}
```
it can be run with:
```bash
python -m llm_eval.generator url generic -o custom.jsonl --dataset my-numbers --ds-cache-dir /my-datasets
```

### Replay

Allows to replay response file from other tasks against a different model.
This is different from re-running same task against different models.
It issues same prompt, but for generation is supplies 'force_generation' argument,
which ensures that generated outputs are also identical.
Differece will be in log probabilities, it's used in 'Divergence Test' below.

```bash
python -m llm_eval.generator url replay -o replayed.jsonl  -f fastchat.jsonl
```


## Complex Tasks

### Divergence Test

This test is done by running sample prompts against 'control' model
and then replaying control model responses against 'test' model.

The following example uses 'generic' task to run 'fastchat' dataset against fp16 model
and then replays responses against fp8 model:
```bash
python -m llm_eval.generator url generic -o fp16.jsonl --max-tokens 256 --dataset fastchat
python -m llm_eval.generator url replay -o fp8.jsonl --max-tokens 256 -f fp16.jsonl
python -m llm_eval.analyzer diff fp16.jsonl fp8.jsonl
```


## Analyzing results

### Single Task

Allows to re-play analysis, which is perfomed after running any simple task:
```bash
python -m llm_eval.analyzer analyzer fastchat.jsonl --task generic
```

### Divergence
```bash
python -m llm_eval.analyzer diff fp16.jsonl fp8.jsonl
```
