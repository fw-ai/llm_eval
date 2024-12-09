from typing import Iterator

from .common import (
    Task,
    Prompt,
    analyze_multichoice,
    load_template,
    load_cacheable_df,
)


class MmluTask(Task):
    CATEGORIES = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    CHOICES = ["A", "B", "C", "D"]

    def __init__(self, ds_cache_dir: str, prompt_style: str = "base"):
        self.prompt_style = prompt_style
        self.ds_cache_dir = ds_cache_dir

    def get_prompts(self) -> Iterator[Prompt]:
        template = load_template("mmlu", self.prompt_style)
        shots_df = load_cacheable_df(
            name="mmlu",
            ds_cache_dir=self.ds_cache_dir,
            hf_ds_name="hails/mmlu_no_train",
            hf_config="all",
            hf_split="dev",
        )
        task_df = load_cacheable_df(
            name="mmlu",
            ds_cache_dir=self.ds_cache_dir,
            hf_ds_name="hails/mmlu_no_train",
            hf_config="all",
            hf_split="test",
        )
        for cat in sorted(self.CATEGORIES):
            shots = []
            data = {"shots": shots}

            for _, shot in shots_df.loc[shots_df["subject"] == cat].iterrows():
                shots.append(shot)

            for _, task in task_df.loc[task_df["subject"] == cat].iterrows():
                correct_answer = self.CHOICES[int(task["answer"])]
                data["task"] = task
                prompt = template.render(data)
                yield Prompt(prompt, labels={"correct_answer": correct_answer})

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        analyze_multichoice(responses, verbose=verbose)

    @property
    def max_tokens(self):
        return 1
