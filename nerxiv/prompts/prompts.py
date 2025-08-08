from pydantic import BaseModel, Field


class Example(BaseModel):
    """
    Represents an example for a prompt, containing input text and expected output.
    """

    input: str = Field(..., description="Input text for the prompt.")
    output: str = Field(..., description="Expected output from the prompt.")


class Prompt(BaseModel):
    """
    Represents a prompt object with various fields to define its structure and content. The final prompt
    is built using the `build()` method, which formats the prompt based on the provided text and the fields defined in this class.
    """

    # expertise fields
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
    constraints: list[str] = Field(
        [],
        description="""
        Constraints to follow in the output of the prompt. These are formatted as 'Important constaints when generating the output: `constraints`'.
        They are also instructions to avoid unused text, broken formats or sentences, etc.
        """,
    )

    # example fields
    examples: list[Example] = Field(
        [],
        description="""
        Examples to illustrate the prompt. These are formatted as 'Examples of how to answer the prompt:
        Example 1:
            - Input text: `example.input`
            - Answer: `example.output`'
        The examples are used to guide the model on how to answer the prompt.
        """,
    )

    # structured schema fields
    # output_schema: BaseModel | None = Field(
    #     None, description="The expected output schema for the prompt."
    # )
    # target_field: list[str] = Field([], description="Fields to target in the output.")

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
            expert_lines = f"You are a {self.expert} assistant"
            if self.sub_field_expertise:
                expert_lines = (
                    f"{expert_lines} with expertise in {self.sub_field_expertise}."
                )
            lines.append(expert_lines)

        # Instructions
        if self.main_instruction:
            instruction_lines = f"Given the following scientific text, your task is: {self.main_instruction}"
            if self.secondary_instructions:
                instruction_lines = f"{instruction_lines}.\nAdditionally, you also need to follow these instructions:"
                for sec_instruction in self.secondary_instructions:
                    instruction_lines += f"\n{sec_instruction}"
            lines.append(instruction_lines)

        # Constraints
        if self.constraints:
            constraint_lines = "Important constaints when generating the output:"
            for constraint in self.constraints:
                constraint_lines += f"\n- {constraint}"
            lines.append(constraint_lines)

        # Examples
        if self.examples:
            example_lines = "Examples of how to answer the prompt:"
            for i, example in enumerate(self.examples):
                example_lines += f"\nExample {i}:\n- Input text: {example.input}\n  Answer: {example.output}"
            lines.append(example_lines)

        # Structured schema
        # TODO add

        # Appending text
        lines.append(f"\nText:\n{text}")
        return "\n".join(lines)


class PromptRegistryEntry(BaseModel):
    """
    Represents a registry entry for a prompt, containing the retriever query and the prompt itself. This
    is used to register prompts in the `QUERY_REGISTRY` and `RETRIEVER_QUERY_REGISTRY`.
    """

    retriever_query: str = Field(..., description="The query used in the retriever.")
    prompt: Prompt = Field(..., description="The prompt to use for the query.")
