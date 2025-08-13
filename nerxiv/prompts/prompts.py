from typing import Any

from pydantic import BaseModel, Field, model_validator

from nerxiv.utils import clean_description


class Example(BaseModel):
    """
    Represents an example for a prompt, containing input text and expected output.
    """

    input: str = Field(..., description="Input text for the prompt.")
    output: str = Field(..., description="Expected output from the prompt.")


class BasePrompt(BaseModel):
    """
    Base class used as an interface for other prompt classes. It defines the common fields and methods
    that all prompts should implement. This class is not meant to be instantiated directly.
    """

    expert: str = Field(
        ...,
        description="""
        The expert or main field of expertise for the prompt. For example, 'Condensed Matter Physics'.
        """,
    )

    sub_field_expertise: str | None = Field(
        None,
        description="""
        The sub-field of expertise for the prompt. For example, 'many-body physics simulations'.
        """,
    )

    examples: list[Example] = Field(
        [],
        description="""
        Examples to illustrate the prompt. These are formatted as:

        'Examples of how to answer the prompt:
        Example 1:
            - Input text: `example.input`
            - Answer: `example.output`'

        They are used to guide the model on how to answer the prompt.
        """,
    )

    constraints: list[str] = Field(
        [],
        description="""
        Constraints to be followed in the output of the prompt. These are formatted as

        'Important constraints when generating the output: `constraints`'.

        They are mainly used as instructions to avoid unused text, broken formats or sentences, etc.
        """,
    )

    def _build_intro(self) -> str:
        """
        Builds the introduction for the prompt, which includes the `expert` and `sub_field_expertise` of the LLM.

        Returns:
            str: The formatted introduction string.
        """
        expert_lines = f"You are a {self.expert} assistant"
        if self.sub_field_expertise:
            expert_lines = (
                f"{expert_lines} with expertise in {self.sub_field_expertise}"
            )
        return expert_lines

    def _build_examples(self) -> str:
        """
        Builds the examples for the prompt, which illustrate how to answer the prompt.

        Returns:
            str: The formatted examples string.
        """
        example_lines = "Examples of how to answer the prompt:"
        for i, example in enumerate(self.examples):
            example_lines += f"\nExample {i + 1}:\n- Input text: {example.input}\n  Answer: {example.output}"
        return example_lines

    def _build_constraints(self) -> str:
        """
        Builds the constraints for the prompt, which are important instructions to follow when generating the output.

        Returns:
            str: The formatted constraints string.
        """
        constraint_lines = "Important constraints when generating the output:"
        for constraint in self.constraints:
            constraint_lines += f"\n- {constraint}"
        return constraint_lines


class Prompt(BasePrompt):
    """
    Represents a prompt object with various fields to define its structure and content. The final prompt
    is built using the `build()` method, which formats the prompt based on the provided text and the fields defined in this class.
    """

    # instruction fields
    main_instruction: str = Field(
        ...,
        description="""
        Main instruction for the prompt. This has to be written in the imperative form, e.g. 'identify all mentions of the system being simulated'.
        The format in the prompt is "Given the following scientific text, your task is `main_instruction`",
        """,
    )

    secondary_instructions: list[str] = Field(
        [],
        description="""
        Secondary instructions for the prompt. These are additional instructions that complement `main_instruction`
        and are formatted as "Additionally, you also need to follow these instructions: `secondary_instructions`".
        """,
    )

    def _build_instructions(self) -> str:
        """
        Builds the instructions for the prompt using the `main_instruction` and `secondary_instructions` fields. This is
        used to format the instructions that will be sent to the LLM for generation.

        Returns:
            str: The formatted instructions string.
        """
        instruction_lines = f"Given the following scientific text, your task is: {self.main_instruction}"
        if self.secondary_instructions:
            instruction_lines = f"{instruction_lines}\nAdditionally, you also need to follow these instructions:"
            for sec_instruction in self.secondary_instructions:
                instruction_lines += f"\n- {sec_instruction}"
        return instruction_lines

    def build(self, text: str) -> str:
        """
        Builds the prompt based on the fields defined in this class. This is used to format the prompt
        and append the `text` to be sent to the LLM for generation.

        Args:
            text (str): The text to append to the prompt.

        Returns:
            str: The formatted prompt ready to be sent to the LLM.
        """
        lines = []

        # Expertise lines
        if self.expert:
            lines.append(self._build_intro())

        # Instructions
        if self.main_instruction:
            lines.append(self._build_instructions())

        # Constraints
        if self.constraints:
            lines.append(self._build_constraints())

        # Examples
        if self.examples:
            lines.append(self._build_examples())

        # Appending text
        lines.append(f"\nText:\n{text}")
        return "\n".join(lines)


class StructuredPrompt(BasePrompt):
    """
    Represents a prompt object with various fields to define its structure and content. The final prompt
    is built using the `build()` method, which formats the prompt based on the provided text and the fields defined in this class.

    **Note**: The main difference with the `Prompt` class is that `StructuredPrompt` is designed to work with a specific output schema,
    so instead of using `main_instructions`, `secondary_instructions` and `constraints`, the instructions are automatically defined by `output_schema`
    and `target_fields`.
    """

    output_schema: type[BaseModel] = Field(
        ...,
        description="""
        The target `BaseModel` schema in which the fields to be extracted are defined.
        """,
    )

    target_fields: list[str] = Field(
        ...,
        description="""
        The fields within `output_schema` that the prompt should extract.
        """,
    )

    @model_validator(mode="after")
    @classmethod
    def validate_target_fields_in_schema(cls, data: Any) -> Any:
        """
        Validates that the `target_fields` are defined in the `output_schema` and that they are of type `Field`.

        Args:
            data (Any): The data containing the fields values to validate.

        Returns:
            Any: The data with the validated fields.
        """
        model_properties = data.output_schema.model_json_schema().get("properties", {})
        for field in data.target_fields:
            if field not in model_properties:
                raise ValueError(
                    f"Field '{field}' is not defined in the output schema '{data.output_schema.__name__}'."
                )
        return data

    def _build_instructions(self) -> str:
        """
        Builds the instructions for the prompt using the `output_schema` and `target_fields` fields. This is
        used to format the instructions that will be sent to the LLM for generation.

        Returns:
            str: The formatted instructions string.
        """
        # gets the model schema metadata as a dictionary
        model = self.output_schema.model_json_schema()

        # name and description of the class which inherits from BaseModel
        name = model.get("title")
        description = clean_description(
            model.get("description", "<<no definition provided>>")
        )
        instruction_lines = f"Given the following scientific text, your task is: to identify all mentions of the {name}. "
        instruction_lines += f"This is defined as {description}. "

        instruction_lines += "You must extract the values of the following fields:"
        # getting the fields defined for the class and maching them with `target_fields`
        properties = model.get("properties", {})
        for field in self.target_fields:
            prop = properties.get(field, {})
            prop_description = clean_description(prop.get("description"))
            prop_type = prop.get("type")
            if not prop_type:
                prop_types = [
                    p.get("type")
                    for p in prop.get("anyOf", [])
                    if p.get("type") != "null"
                ]  # only non-null types
                prop_type = prop_types[0]
            instruction_lines += f"\n- {field} defined as '{prop_description}' and which is of type {prop_type}"
            # TODO add data type

        instruction_lines += (
            "\nYou must return the extracted values in the following format:"
            "\n{" + f"'{name}': " + "{"
        )
        for field in self.target_fields:
            instruction_lines += f"'{field}': <parsed-value>, "

        instruction_lines += "} }"
        return instruction_lines

    def build(self, text: str) -> str:
        """
        Builds the prompt based on the fields defined in this class. This is used to format the prompt
        and append the `text` to be sent to the LLM for generation.

        Args:
            text (str): The text to append to the prompt.

        Returns:
            str: The formatted prompt ready to be sent to the LLM.
        """
        lines = []

        # Expertise lines
        if self.expert:
            lines.append(self._build_intro())

        # Instructions
        lines.append(self._build_instructions())

        # Constraints
        if self.constraints:
            lines.append(self._build_constraints())

        # Examples
        if self.examples:
            lines.append(self._build_examples())

        # Appending text
        lines.append(f"\nText:\n{text}")
        return "\n".join(lines)


class PromptRegistryEntry(BaseModel):
    """
    Represents a registry entry for a prompt, containing the retriever query and the prompt itself. This
    is used to register prompts in the `QUERY_REGISTRY` and `RETRIEVER_QUERY_REGISTRY`.
    """

    retriever_query: str = Field(..., description="The query used in the retriever.")
    prompt: BasePrompt = Field(..., description="The prompt to use for the query.")
