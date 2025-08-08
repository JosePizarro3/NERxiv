import pytest

from nerxiv.prompts.prompts import Example, Prompt


class TestPrompt:
    @pytest.mark.parametrize(
        "sub_field_expertise, secondary_instructions, constraints, examples, result",
        [
            (
                None,
                [],
                [],
                [],
                "You are a condensed matter physics assistant\nGiven the following scientific text, your task is: identify the acronyms of all methods used\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
            ),
            (
                "quantum materials",
                [],
                [],
                [],
                "You are a condensed matter physics assistant with expertise in quantum materials\nGiven the following scientific text, your task is: identify the acronyms of all methods used\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
            ),
            (
                "quantum materials",
                [
                    "return only the acronyms",
                    "multiple methods are allowed and should be separated by commas",
                ],
                ["no yapping"],
                [],
                "You are a condensed matter physics assistant with expertise in quantum materials\nGiven the following scientific text, your task is: identify the acronyms of all methods used\nAdditionally, you also need to follow these instructions:\n- return only the acronyms\n- multiple methods are allowed and should be separated by commas\nImportant constaints when generating the output:\n- no yapping\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
            ),
            (
                "quantum materials",
                [
                    "return only the acronyms",
                    "multiple methods are allowed and should be separated by commas",
                ],
                ["no yapping"],
                [
                    Example(input="We work using DFT.", output="DFT"),
                    Example(input="We used DFT and also DMFT.", output="DFT,DMFT"),
                ],
                "You are a condensed matter physics assistant with expertise in quantum materials\nGiven the following scientific text, your task is: identify the acronyms of all methods used\nAdditionally, you also need to follow these instructions:\n- return only the acronyms\n- multiple methods are allowed and should be separated by commas\nImportant constaints when generating the output:\n- no yapping\nExamples of how to answer the prompt:\nExample 1:\n- Input text: We work using DFT.\n  Answer: DFT\nExample 2:\n- Input text: We used DFT and also DMFT.\n  Answer: DFT,DMFT\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
            ),
        ],
    )
    def test_build(
        self,
        sub_field_expertise: str,
        secondary_instructions: list[str],
        constraints: list[str],
        examples: list[Example],
        result: str,
    ):
        prompt = Prompt(
            expert="condensed matter physics",
            sub_field_expertise=sub_field_expertise,
            main_instruction="identify the acronyms of all methods used",
            secondary_instructions=secondary_instructions,
            constraints=constraints,
            examples=examples,
        )
        prompt_text = prompt.build(
            text="The simulations were performed using DFT and DMFT methods."
        )
        assert prompt_text == result
