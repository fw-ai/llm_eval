import argparse
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import itertools
import math

import numpy
from tabulate import tabulate

from .tasks import get_task_class


@dataclass
class DivergenceStats:
    matches: int = 0
    mismatches: int = 0
    kld: float = 0.0
    beta: float = 0.0

    def add(self, other: "DivergenceStats"):
        self.matches += other.matches
        self.mismatches += other.mismatches
        self.kld += other.kld
        self.beta += other.beta

    def avg(self) -> "DivergenceStats":
        num = self.matches + self.mismatches
        return DivergenceStats(
            matches=self.matches,
            mismatches=self.mismatches,
            kld=self.kld / num if num else 0,
            beta=self.beta / num if num else 0,
        )


@dataclass
class ResponseDiff:
    pos: int = 0
    id_a: str = ""
    id_b: str = ""
    prefill: DivergenceStats = field(default_factory=DivergenceStats)
    gen: DivergenceStats = field(default_factory=DivergenceStats)


@dataclass
class Diff:
    matched_tokens: int = 0
    mismatched_tokens: int = 0
    resp: List[ResponseDiff] = field(default_factory=list)
    prefill: DivergenceStats = field(default_factory=DivergenceStats)
    gen: DivergenceStats = field(default_factory=DivergenceStats)


@dataclass
class TokenInfos:
    tokens: List[str]
    lps: List[float]
    top_lps: List[Dict[str, float]] | None


def _compute_diff(
    responses_a: List, responses_b: List, verbose_mismatch: bool = False
) -> Diff:
    assert len(responses_a) == len(responses_b), "Different number of responses"

    diff = Diff()
    for i, (resp_a, resp_b) in enumerate(zip(responses_a, responses_b)):
        resp_diff = ResponseDiff(pos=i, id_a=resp_a["id"], id_b=resp_b["id"])

        pf_a, gen_a = _process_response(resp_a)
        pf_b, gen_b = _process_response(resp_b)
        # Assert that responses have identical prompts and generation lengths.
        assert (pf_a is None) == (pf_b is None), "Different prompt"
        if pf_a is not None:
            assert pf_a.tokens == pf_b.tokens, "Different prompt"
        assert len(gen_a.tokens) == len(gen_b.tokens), "Different number of generated tokens"

        if pf_a.top_lps is not None and pf_b.top_lps is not None:
            for top_lp_a, top_lp_b in zip(pf_a.top_lps, pf_b.top_lps):
                resp_diff.prefill.add(_compute_divergence(top_lp_a, top_lp_b))
        if gen_a.top_lps is not None and gen_b.top_lps is not None:
            for top_lp_a, top_lp_b in zip(gen_a.top_lps, gen_b.top_lps):
                resp_diff.gen.add(_compute_divergence(top_lp_a, top_lp_b))

        context = ""
        if verbose_mismatch:
            for j, (token_a, token_b, l_a, l_b) in enumerate(
                itertools.zip_longest(gen_a.tokens, gen_b.tokens, gen_a.lps, gen_b.lps)
            ):
                if token_a != token_b:
                    print(
                        f"================================= {resp_a['id']} vs {resp_b['id']} at token {j}"
                    )
                    print(
                        f"{context[-100:]}| `{token_a}` vs `{token_b}`, logprobs: {l_a} vs {l_b}"
                    )
                context += token_a or token_b or ""

        diff.resp.append(resp_diff)

    for resp_diff in diff.resp:
        diff.prefill.add(resp_diff.prefill)
        resp_diff.prefill = resp_diff.prefill.avg()

        diff.gen.add(resp_diff.gen)
        resp_diff.gen = resp_diff.gen.avg()
    diff.prefill = diff.prefill.avg()
    diff.gen = diff.gen.avg()

    return diff


def _process_response(resp) -> Tuple[TokenInfos | None, TokenInfos]:
    choice = resp["choices"][0]
    logprobs = choice.get("logprobs")
    assert logprobs is not None
    tokens = logprobs["tokens"]
    lps = logprobs["token_logprobs"]
    top_lps = logprobs.get("top_logprobs")

    pf_len = resp["usage"]["prompt_tokens"]
    gen_len = resp["usage"]["completion_tokens"]
    if len(tokens) == gen_len + pf_len:
        pf_tokens = tokens[:pf_len]
        pf_lps = lps[:pf_len]
        if top_lps is not None:
            pf_top_lps = top_lps[:pf_len]
        else:
            pf_top_lps = None
        pf_info = TokenInfos(pf_tokens, pf_lps, pf_top_lps)
        gen_offset = pf_len
    else:
        pf_info = None
        gen_offset = 0
    gen_tokens = tokens[gen_offset:]
    gen_lps = lps[gen_offset:]
    if top_lps is not None:
        gen_top_lps = top_lps[gen_offset:]
    else:
        top_lps = None
    gen_info = TokenInfos(gen_tokens, gen_lps, gen_top_lps)

    return pf_info, gen_info


def _compute_divergence(
    a_top_lps: Dict[str, float], b_top_lps: Dict[str, float]
) -> DivergenceStats:
    if a_top_lps and b_top_lps and next(iter(a_top_lps)) == next(iter(b_top_lps)):
        matches = 1
    else:
        matches = 0

    common_tokens = set(a_top_lps.keys()).intersection(b_top_lps.keys())

    p = _construct_approx_distribution(common_tokens, a_top_lps)
    q = _construct_approx_distribution(common_tokens, b_top_lps)

    return DivergenceStats(
        matches=matches,
        mismatches=1 - matches,
        kld=_approximate_kld(p, q),
        beta=_approximate_beta(p, q),
    )


def _construct_approx_distribution(common_tokens, top_tokens):
    prob_dist = [0.0] * (len(common_tokens))
    for i, k in enumerate(common_tokens):
        prob_dist[i] = math.exp(top_tokens[k])
    return prob_dist


def _approximate_kld(approx_dist_p, approx_dist_q):
    normalized_leftover_p = 1 - sum(approx_dist_p)
    normalized_leftover_q = 1 - sum(approx_dist_q)

    p_np = numpy.asarray(approx_dist_p)
    q_np = numpy.asarray(approx_dist_q)

    kld_of_approx = numpy.sum(p_np * numpy.log(p_np / q_np))

    if normalized_leftover_q > 1e-7 and normalized_leftover_p > 1e-7:
        kld_of_approx += normalized_leftover_p * numpy.log(
            normalized_leftover_p / normalized_leftover_q
        )
    return kld_of_approx


def _approximate_beta(approx_dist_p, approx_dist_q):
    normalized_leftover_p = 1 - sum(approx_dist_p)
    normalized_leftover_q = 1 - sum(approx_dist_q)
    p_np = numpy.asarray(approx_dist_p)
    q_np = numpy.asarray(approx_dist_q)
    return 1 - (
        numpy.minimum(p_np, q_np).sum()
        + numpy.minimum(normalized_leftover_p, normalized_leftover_q)
    )


def _load_responses(fn, args):
    ret = []
    with open(fn) as f:
        for i, resp in enumerate(f):
            if args.limit and i >= args.limit:
                break
            resp = json.loads(resp)
            ret.append(resp)
    return ret


def _run_diff_command(args):
    responses_a = _load_responses(args.diff_a, args)
    responses_b = _load_responses(args.diff_b, args)

    diff = _compute_diff(responses_a, responses_b, args.verbose_mismatch)
    table = [
        ["Prefill KLD", diff.prefill.kld],
        ["Prefill Beta", diff.prefill.beta],
        ["Gen KLD", diff.gen.kld],
        ["Gen Beta", diff.gen.beta],
        ["Gen Top Token Matches", diff.gen.matches],
        ["Gen Top Token Mismatches", diff.gen.mismatches],
    ]
    print(tabulate(table, tablefmt="plain"))
    if args.verbose:
        table = [
            [
                "Pos",
                "ID A",
                "ID B",
                "Prefill KLD",
                "Prefill Beta",
                "Gen KLD",
                "Gen Beta",
                "Gen Top Token Matches",
                "Gen Top Token Mismatches",
            ]
        ]
        for resp_diff in diff.resp:
            table.append(
                [
                    resp_diff.pos,
                    resp_diff.id_a,
                    resp_diff.id_b,
                    resp_diff.prefill.kld,
                    resp_diff.prefill.beta,
                    resp_diff.gen.kld,
                    resp_diff.gen.beta,
                    resp_diff.gen.matches,
                    resp_diff.gen.mismatches,
                ]
            )
        print(tabulate(table))


def _run_analyze_command(args):
    responses = _load_responses(args.responses, args)
    task = get_task_class(args.task)
    task.analyze(responses, verbose=args.verbose)


def main():
    parser = argparse.ArgumentParser(prog="eval-differ")

    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser(
        "diff", help="Compute diff between two responses.jsonl"
    )
    diff_parser.add_argument(
        "diff_a", type=str, help="Path to the first responses.jsonl"
    )
    diff_parser.add_argument(
        "diff_b", type=str, help="Path to the second responses.jsonl"
    )
    diff_parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction)
    diff_parser.add_argument(
        "-V", "--verbose-mismatch", action=argparse.BooleanOptionalAction
    )
    diff_parser.add_argument("--limit", "-l", type=int)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze responses.jsonl")
    analyze_parser.add_argument(
        "responses", type=str, help="Path to the responses.jsonl"
    )
    analyze_parser.add_argument(
        "-v", "--verbose", action=argparse.BooleanOptionalAction
    )
    analyze_parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="generic"
    )
    analyze_parser.add_argument("--limit", "-l", type=int)

    args = parser.parse_args()

    if args.command == "diff":
        _run_diff_command(args)
    elif args.command == "analyze":
        _run_analyze_command(args)


if __name__ == "__main__":
    main()
