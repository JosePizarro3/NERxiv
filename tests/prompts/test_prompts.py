import pytest

from nerxiv.datamodel.crystal_structure import ChemicalFormulation
from nerxiv.prompts.prompts import BasePrompt, Example, Prompt, StructuredPrompt


class TestBasePrompt:
    @pytest.mark.parametrize(
        "sub_field_expertise, result",
        [
            (None, "You are a condensed matter physics assistant"),
            ("", "You are a condensed matter physics assistant"),
            (
                "quantum materials",
                "You are a condensed matter physics assistant with expertise in quantum materials",
            ),
        ],
    )
    def test_build_intro(self, sub_field_expertise: str | None, result: str):
        """Test the `_build_intro` method of BasePrompt."""
        prompt = BasePrompt(
            expert="condensed matter physics",
            sub_field_expertise=sub_field_expertise,
        )
        assert prompt._build_intro() == result

    @pytest.mark.parametrize(
        "examples, result",
        [
            ([], "Examples of how to answer the prompt:"),
            (
                [
                    Example(input="We work using DFT.", output="DFT"),
                    Example(input="We used DFT and also DMFT.", output="DFT,DMFT"),
                ],
                "Examples of how to answer the prompt:\nExample 1:\n- Input text: We work using DFT.\n  Answer: DFT\nExample 2:\n- Input text: We used DFT and also DMFT.\n  Answer: DFT,DMFT",
            ),
        ],
    )
    def test_build_examples(self, examples: list[str], result: str):
        """Test the `_build_examples` method of BasePrompt."""
        prompt = BasePrompt(
            expert="condensed matter physics",
            examples=examples,
        )
        assert prompt._build_examples() == result

    @pytest.mark.parametrize(
        "constraints, result",
        [
            ([], "Important constraints when generating the output:"),
            (
                ["no yapping"],
                "Important constraints when generating the output:\n- no yapping",
            ),
        ],
    )
    def test_build_constraints(self, constraints: list[str], result: str):
        """Test the `_build_constraints` method of BasePrompt."""
        prompt = BasePrompt(
            expert="condensed matter physics",
            constraints=constraints,
        )
        assert prompt._build_constraints() == result


class TestPrompt:
    @pytest.mark.parametrize(
        "secondary_instructions, result",
        [
            (
                [],
                "Given the following scientific text, your task is: identify the acronyms of all methods used",
            ),
            (
                [
                    "return only the acronyms",
                    "multiple methods are allowed and should be separated by a comma",
                ],
                "Given the following scientific text, your task is: identify the acronyms of all methods used\nAdditionally, you also need to follow these instructions:\n- return only the acronyms\n- multiple methods are allowed and should be separated by a comma",
            ),
        ],
    )
    def test_build_instructions(self, secondary_instructions: list[str], result: str):
        """Test the `_build_instructions` method of BasePrompt."""
        prompt = Prompt(
            expert="condensed matter physics",
            main_instruction="identify the acronyms of all methods used",
            secondary_instructions=secondary_instructions,
        )
        assert prompt._build_instructions() == result

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
                "You are a condensed matter physics assistant with expertise in quantum materials\nGiven the following scientific text, your task is: identify the acronyms of all methods used\nAdditionally, you also need to follow these instructions:\n- return only the acronyms\n- multiple methods are allowed and should be separated by commas\nImportant constraints when generating the output:\n- no yapping\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
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
                "You are a condensed matter physics assistant with expertise in quantum materials\nGiven the following scientific text, your task is: identify the acronyms of all methods used\nAdditionally, you also need to follow these instructions:\n- return only the acronyms\n- multiple methods are allowed and should be separated by commas\nImportant constraints when generating the output:\n- no yapping\nExamples of how to answer the prompt:\nExample 1:\n- Input text: We work using DFT.\n  Answer: DFT\nExample 2:\n- Input text: We used DFT and also DMFT.\n  Answer: DFT,DMFT\n\nText:\nThe simulations were performed using DFT and DMFT methods.",
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
        """Test the `build` method of Prompt."""
        prompt = Prompt(
            expert="condensed matter physics",
            sub_field_expertise=sub_field_expertise,
            main_instruction="identify the acronyms of all methods used",
            secondary_instructions=secondary_instructions,
            constraints=constraints,
            examples=examples,
        )
        assert (
            prompt.build(
                text="The simulations were performed using DFT and DMFT methods."
            )
            == result
        )


class TestStructuredPrompt:
    def test_build_instructions(self):
        """Test the `_build_instructions` method of StructuredPrompt."""
        prompt = StructuredPrompt(
            expert="condensed matter physics",
            output_schema=ChemicalFormulation,
            target_fields=["iupac", "reduced"],
        )
        assert prompt._build_instructions() == (
            "Given the following scientific text, your task is: to identify all mentions of "
            "the ChemicalFormulation section. This is defined as a A ChemicalFormulation is a descriptive "
            "representation of the chemical composition of a material system, expressed in one or "
            "more standardized formula formats (e.g., IUPAC, anonymous, Hill, or reduced), each "
            "encoding the stoichiometry and elemental ordering according to specific conventions. "
            "For the compound H2O2 (hydrogen peroxide), the different formulations would be: iupac: "
            "H2O2 anonymous: AB hill: H2O2 reduced: H2O2 You must extract the values of the following "
            "fields:\n- iupac defined as Chemical formula where the elements are ordered using a "
            "formal list based on electronegativity as defined in the IUPAC nomenclature of inorganic "
            "chemistry (2005): - https://en.wikipedia.org/wiki/List_of_inorganic_compounds Contains "
            "reduced integer chemical proportion numbers where the proportion number is omitted if it "
            "is 1. and which is of type string\n- reduced defined as Alphabetically sorted chemical "
            "formula with reduced integer chemical proportion numbers. The proportion number is omitted "
            "if it is 1. and which is of type string\nYou must return the extracted values in JSON "
            "format:\n```json\n{\n  'ChemicalFormulation': {\n    'iupac': <parsed-value>,\n    'reduced': <parsed-value>,\n  }\n}\n```\n"
            "Note that <parsed-value> means a value of the correct type defined for that field."
        )
