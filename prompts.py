# -*- coding: utf-8 -*-
"""Prompt templates for earnings call processing (parent chunks, subchunks, themes)."""
from langchain_core.prompts import PromptTemplate


class PromptCollections:
    def __init__(self):
        pass

    def get_langchain_supported_prompt(self, input_variables: list, template_string: str) -> PromptTemplate:
        return PromptTemplate(input_variables=input_variables, template=template_string)

    def get_context_summary(self, i: str = "F") -> str:
        """Alias for get_parent_chunks for compatibility. i in ('F','L','M')."""
        return self.get_parent_chunks(i)

    def get_parent_chunks(self, i: str = "F") -> str:
        if i == "M":
            return """
                You are a financial analyst summarizing an earnings call transcript.
                Summarize the **current section** below in a concise, professional analyst tone.
                Use the **Previous and Next section** only to clarify ambiguous references in the current section,
                not to summarize it.
                Guidelines:
                - Focus on the main ideas, developments, and management commentary relevant to the business or financial outlook.
                - Highlight key financial, operational, strategic, or market-related points — but feel free to include any other insights that add meaningful context.
                - If the section is qualitative, describe the underlying message or tone instead of stating that no figures were provided.
                - Avoid any meta phrasing such as "Summary", "this section", or "no new data".
                - Write naturally, in 3–5 sentences, as you would in an analyst note — factual, concise, and context-aware.
                Previous section (for context only):
                {prevDoc}
                Current section:
                {doc}
                Next section (for context only):
                {nxtDoc}
            """
        elif i == "L":
            return """
                You are a financial analyst summarizing an earnings call transcript.
                Summarize the **current section** below in a concise, professional analyst tone.
                Use the **Previous section** only to clarify ambiguous references in the current section,
                not to summarize it.
                Guidelines:
                - Focus on the main ideas, developments, and management commentary relevant to the business or financial outlook.
                - Highlight key financial, operational, strategic, or market-related points — but feel free to include any other insights that add meaningful context.
                - If the section is qualitative, describe the underlying message or tone instead of stating that no figures were provided.
                - Avoid any meta phrasing such as "Summary", "this section", or "no new data".
                - Write naturally, in 3–5 sentences, as you would in an analyst note — factual, concise, and context-aware.
                Previous section (for context only):
                {prevDoc}
                Current section:
                {doc}
            """
        else:  # F
            return """
                You are a financial analyst summarizing an earnings call transcript.
                Summarize the **current section** below in a concise, professional analyst tone.
                Use the **next section** only to clarify ambiguous references in the current section,
                not to summarize it.
                Guidelines:
                - Focus on the main ideas, developments, and management commentary relevant to the business or financial outlook.
                - Highlight key financial, operational, strategic, or market-related points — but feel free to include any other insights that add meaningful context.
                - If the section is qualitative, describe the underlying message or tone instead of stating that no figures were provided.
                - Avoid any meta phrasing such as "Summary", "this section", or "no new data".
                - Write naturally, in 3–5 sentences, as you would in an analyst note — factual, concise, and context-aware.
                Current section:
                {doc}
                Next section (for context only):
                {nxtDoc}
            """

    def get_subchunks(self) -> str:
        return """
            You are an **expert** Financial Analyst AI tasked with summarizing sections of earnings transcript text.
            ---------------
            **Current section** to summarize:
            {doc}
            ---------------
            Your goal is to meticulously analyze the "Current section" and extract key insights. Follow these steps:
            1.  **Identify Distinct Themes:**
                * Carefully read the "Current section" and identify all **distinct financial or operational themes**.
                * A "theme" is a significant topic of discussion related to financial performance and should represent a **substantive piece of information** for an investor.
                * Do not invent themes not explicitly discussed.
                * Crucially, group closely related points under a single unified theme to avoid fragmentation.
            2.  **Summarize Each Theme:**
                * For each identified theme, provide a **concise, factual summary** based *only* on information found within the "Current section."
                * Focus on extracting specific metrics, figures, C-suite commentary, and their implications *if stated in the text*.
                * Avoid speculation, personal opinions, or information not explicitly present in the "Current section."
                * Do not include introductory phrases like "The current section discusses..." or "In this section..." within your individual summaries.
            3.  **Assign Sentiment Probabilities:**
                * For each summary point, assign sentiment probabilities reflecting the tone and implication of that specific information in **financial terms for the company**:
                    * `positive`: Reflects beneficial outcomes, growth, exceeding expectations, efficiency gains, or confidence-building commentary related to the company's performance or outlook.
                    * `negative`: Reflects risks, concerns, underperformance, deteriorating conditions, challenges, or cautionary statements regarding the company.
                    * `neutral`: Reflects factual statements, mixed results, or commentary without a clearly positive or negative financial implication for the company as presented.
                * Ensure the sum of `positive`, `negative`, and `neutral` probabilities is **exactly 1.0** for each summary. Distribute the probabilities based on the nuanced financial tone of the statement; not all summaries will be purely one sentiment.
            4.  **Assign Action Tag:**
                * For each summary, assign a single, most appropriate tag based on the nature of the information presented. Choose one of the following tags:
                    * **E (Executed):** The summary clearly describes actions already taken or outcomes already realized by the company (e.g., investments made, contracts secured, deliveries completed, systems deployed).
                    * **P (Planned):** The summary refers to the company's future plans, intentions, projections, or expectations (e.g., upcoming launches, projected revenues, anticipated growth, future strategies).
                    * **N (NA):** The summary does not clearly describe either executed actions or planned actions, or lacks sufficient context for classification.
                    * If uncertain whether a statement is executed or planned, prefer "N (NA)" to avoid misclassification.
            5.  **Extract Financial Keywords:**
                * For each summary, list **3–8 short keywords or short phrases** that are financially relevant and sound (e.g. revenue growth, capex, margin, guidance, EBITDA, market share, debt, dividend).
                * Use terms an analyst or investor would use. Only include terms that appear or are clearly implied in the "Current section."
            
            **Output Format:**
            Return the output **only as a JSON list**. Each item in the list must strictly adhere to this structure:
            [
                {{
                    "childChunk": "<concise summary of one theme from the Current section>",
                    "positive": <float>,
                    "negative": <float>,
                    "neutral": <float>,
                    "tag": "<E, P, or N>",
                    "keywords": ["<keyword1>", "<keyword2>", ...]
                }}
            ]
            **Important Considerations:**
            * Only include financially or operationally **meaningful points** directly from the "Current section."
            * Maintain objectivity and stick strictly to the information provided in the "Current section" for the content of your summaries.
            * Aim for a level of summarization that is insightful rather than a granular reiteration of every sentence.
            """

    def get_neg_theme(self) -> str:
        return """
            You are a financial analyst. Your task is to identify and summarize key **themes that reflect genuinely negative developments** — including actions/results that have already materialized and concerns or risks that are expected to impact the company adversely in the future.
            Input data is a list of entries. Each entry contains a `childChunk` which contains a list of negative insights and `parentChunk` which contains the full context from which they were derived.
            ---
            ### Your Tasks:
            1. **Identify Only Genuinely Negative Themes:**
            * Identify and name key themes that capture:
                - **Negative Actions/Results:** Failures, setbacks, or challenges that have already occurred.
                - **Negative Outlook:** Anticipated risks, potential issues, or expected negative developments.
            * Group overlapping issues into a single, unified theme. Avoid creating multiple themes for the same core problem.
            2. **Theme Details:**
            * For each theme, write a concise but **comprehensive and structured summary**.
            * Do not use bullet points or lists within the summary.
            * Ensure the final summary is comprehensive yet concise.
            * Use the `parentChunk` to provide supporting evidence, specific numbers, or more detailed explanations that build upon the negative points found in the `childChunk`.
            3. **Ensure Distinctness:** Make sure each theme is unique. Do not repeat the same information or findings across different themes.
            4. **Avoid Personal Names:** Do **not mention individuals by name**. Generalize such references.
            ---
            ### Output Format:
            Return a list of JSON objects like this:
            [{{"Name": "Theme Name", "Details": "Theme description and insights extracted from the summaries and context."}}]
            ---
            **Input Data:**
            (Each item in the list will have this structure: {{"childChunk": [...], "parentChunk": "..." }})
            {docs}
            """

    def get_pos_theme(self) -> str:
        return """
            You are a financial analyst. Your task is to identify and summarize key **themes that reflect genuinely positive developments** — including actions/results that have already materialized and opportunities or strengths that are expected to impact the company favorably in the future.
            Input data is a list of entries. Each entry contains a `childChunk` which contains a list of positive insights and `parentChunk` which contains the full context from which they were derived.
            ---
            ### Your Tasks:
            1. **Identify Only Genuinely Positive Themes:**
            * Identify and name key themes that capture:
                - **Positive Actions/Results:** Successes, achievements, or strengths that have already occurred.
                - **Positive Outlook:** Anticipated opportunities, potential benefits, or expected positive developments.
            * Group overlapping issues into a single, unified theme.
            2. **Theme Details:**
            * For each theme, write a concise but **comprehensive and structured summary**.
            * Do not use bullet points or lists within the summary.
            * Ensure the final summary is comprehensive yet concise.
            * Use the `parentChunk` to provide supporting evidence, specific numbers, or more detailed explanations that build upon the positive points found in the `childChunk`.
            3. **Ensure Distinctness:** Make sure each theme is unique.
            4. **Avoid Personal Names:** Do **not mention individuals by name**.
            ### Output Format:
            Return a list of JSON objects like this:
            [{{"Name": "Theme Name", "Details": "Theme description and insights extracted from the summaries and context."}}]
            ---
            **Input Data:**
            (Each item in the list will have this structure: {{"childChunk": [...], "parentChunk": "..." }})
            {docs}
            """

    def get_executed_theme(self) -> str:
        return """
            As a highly experienced financial research analyst, your task is to synthesize a series of text chunks from a company's earnings call. The provided chunks are specifically filtered to contain information about 'executed' actions and results, rather than future plans or guidance.
            Your objective is to generate a JSON object containing a concise, professional summary of management's executed actions and achievements.
            ---
            **Instructions:**
            1. Analyze the provided JSON list. Each item has **childChunk** (a list of short theme summaries filtered for executed content) and **parentChunk** (the full transcript context from which they were derived). Use both to extract all relevant executed actions and achievements.
            2. Focus exclusively on **executed actions and achievements**. Do not include any forward-looking statements, future plans, or guidance.
            3. **Mandatory Requirement: Capture all quantitative data.** You must include every relevant number, percentage, basis point change, and monetary figure found in the text. Do not miss out on numbers or generalize them.
            4. Synthesize and combine related points into single, comprehensive sentences.
            5. Format the final output as a JSON object with a single key, 'executed_actions'. The value for this key must be a list of these synthesized sentences.
            **Output Json Example:**
            {{"executed_actions": [list of synthesized sentences]}}
            ---
            **Input Data :**
            (Each item in the list will have this structure: {{"childChunk": [...], "parentChunk": "..." }})
            {docs}
            """

    def get_planned_theme(self) -> str:
        return """
            As a highly experienced financial research analyst, your task is to synthesize a series of text chunks from a company's earnings call. The provided chunks are specifically filtered to contain information about 'planned' actions and future guidance, rather than executed results.
            Your objective is to generate a JSON object containing a concise, professional summary of management's future outlook and strategic plans.
            ---
            **Instructions:**
            1. Analyze the provided JSON list. Each item has **childChunk** (a list of short theme summaries filtered for planned/future content) and **parentChunk** (the full transcript context from which they were derived). Use both to extract all relevant planned actions and guidance.
            2. Focus exclusively on future plans, guidance, and forward-looking statements (e.g., 'expected to,' 'anticipates,' 'will be'). Do not include any past achievements or executed actions.
            3. **Mandatory Requirement: Capture all quantitative data.** You must include every relevant number, percentage, basis point change, and monetary figure found in the text. Do not miss out on numbers or generalize them.
            4. Synthesize and combine related points into single, comprehensive sentences.
            5. Format the final output as a JSON object with a single key, 'planned_actions'. The value for this key must be a list of these synthesized sentences.
            **Output Json Example:**
            {{"planned_actions": [list of synthesized sentences]}}
            ---
            **Input Data :**
            (Each item in the list will have this structure: {{"childChunk": [...], "parentChunk": "..." }})
            {docs}
            """
