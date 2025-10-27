from nerxiv.prompts.prompts import Example, Prompt, PromptRegistryEntry

FILTERS_REGISTRY = {
    "filter_material_formula": PromptRegistryEntry(
        retriever_query="""Identify explicit mentions of the system or material *actually simulated*
        in the study. This includes bulk crystals, molecules, or nanostructures. Exclude toy or
        model systems (e.g., square lattice, Hubbard model, honeycomb lattice) unless they are
        directly parameterized for a real material.""",
        prompt=Prompt(
            expert="Condensed Matter Physics",
            main_instruction="Extract all mentions of the *actual simulated system or material*.",
            secondary_instructions=[
                "Include only systems explicitly simulated or modeled with material-specific parameters.",
                "Exclude toy or abstract models (e.g., Hubbard model on a square lattice).",
                "Exclude materials mentioned for comparison or as examples, unless simulation results are directly presented for them.",
                "Normalize chemical formulas when possible (e.g., La₁₋ₓSrₓNiO₂ to La1-xSrxNiO2).",
            ],
            constraints=[
                "Return ONLY the names or formulas of simulated materials, separated by commas.",
                "Do NOT return commentary, reasoning, or model names.",
            ],
            examples=[
                Example(
                    input="The system is a bulk crystal of silicon, which has a diamond cubic structure.",
                    output="Si2",
                ),
                Example(
                    input="The system is doped La₁₋ₓSrₓNiO₂, for x=0.2.",
                    output="La0.8Sr0.2NiO2",
                ),
                Example(
                    input="We perform DFT calculations on LaNiO3 and its doped variant La0.8Sr0.2NiO3.",
                    output="LaNiO3, La0.8Sr0.2NiO3",
                ),
                Example(
                    input="We analyze a two-band Hubbard model on a square lattice to mimic SrVO3.",
                    output="model",
                ),
                Example(
                    input="The Hubbard model is used to study generic correlated behavior.",
                    output="model",
                ),
                Example(
                    input="Our study focuses on Fe2O3 and Fe2O3.25.",
                    output="Fe2O3, Fe2O3.25",
                ),
                Example(
                    input="Our simulations on the honeycomb lattice represent the electronic structure of graphene.",
                    output="graphene | C",
                ),
            ],
        ),
    ),
    "filter_only_dmft": PromptRegistryEntry(
        retriever_query="""Identify all mentions of the method used in the text. The method can be DMFT, DFT+U, DFT,
        Quantum Monte Carlo, Exact Diagonalization, etc.""",
        prompt=Prompt(
            expert="Condensed Matter Physics",
            main_instruction="""Identify the computational methods *actually applied* in the study.
            Focus on methods used for the simulations described (e.g., DFT, DFT+DMFT, QMC, ED). Exclude
            methods only mentioned as comparisons, background, or references.""",
            secondary_instructions=[
                "Return True only if the text indicates that DMFT (or a related variant: DFT+DMFT, EDMFT, LDA+DMFT) was directly applied.",
                "Return False if DMFT is only mentioned as prior work, comparison, or reference.",
                "Ignore other methods (DFT, DFT+U, QMC, ED, etc.) unless combined with DMFT.",
            ],
            constraints=[
                "Return ONLY a single boolean value: True or False.",
                "Do NOT include explanations, text, or context.",
            ],
            examples=[
                Example(
                    input="We use DFT+DMFT to study the electronic properties of the LaCuO4.",
                    output="True",
                ),
                Example(
                    input="We use DFT to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
                Example(
                    input="We use Quantum Monte Carlo to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
                Example(
                    input="In another material, MnO, the DMFT results are in good agreement with our DFT+U results.",
                    output="False",
                ),
                Example(
                    input="We use DFT+U to study the electronic properties of the LaCuO4.",
                    output="False",
                ),
            ],
        ),
    ),
}
