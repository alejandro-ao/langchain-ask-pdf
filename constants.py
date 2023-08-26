# from chromadb.config import Settings
# # Define the Chroma settings
# CHROMA_SETTINGS = Settings(
#         chroma_db_impl='duckdb+parquet',
#         persist_directory=PERSIST_DIRECTORY,
#         anonymized_telemetry=False
# )

research_prompts = {
    "Summarizing and Analysis": [
        "Summarize the main arguments in this article/abstract:",
        "Identify the key findings and implications of this research paper:",
        "Analyze the strengths and weaknesses of this methodology:",
        "Compare and contrast [theory A] and [theory B] in the context of [field]:",
        "Describe the theoretical framework of this study and how it relates to the findings:"
    ],

    "Research Questions and Gaps": [
        "List potential research questions related to [topic]:",
        "Identify gaps in the literature on [topic]:",
        "Generate a list of research hypotheses related to [topic]:",
        "Identify potential areas for future research in the context of this article/abstract:",
        "Suggest novel applications of [theory/concept] within [field]:"
    ],

    "Methodology and Techniques": [
        "What are the limitations of using [statistical method] in [research context]?",
        "Create a recipe for the methods used in this [paper/thesis]",
        "Suggest interdisciplinary approaches to [research question/problem]:",
        "Explain how [qualitative/quantitative] research methods can be used to address [research question]:",
        "Describe the advantages of using a mixed-methods approach for studying [topic]:",
        "Recommend best practices for data collection and analysis in [field/research context]:"
    ]
}

similarity_search_queries = {
    1: "This paper discusses the impact of",
    2: "The main contribution of this work is",
    3: "This research aims to explore",
    4: "The authors propose a new",
    5: "Our research addresses the issue of",
    6: "The authors present a novel approach for",
    7: "This paper introduces a framework for",
    8: "The proposed method achieves state-of-the-art results in",
    9: "In this study, we investigate",
    10: "Our experiments show that",
    11: "The results indicate that",
    12: "Our analysis reveals insights into",
    13: "Our findings shed light on the importance of",
    14: "This section focuses on the relationship between",
    15: "In this section, we discuss the implications of",
    16: "Based on the data, we conclude that",
    17: "In summary, our study reveals",
    18: "The experimental setup consists of",
    19: "Recent findings suggest",
    20: "The limitations of our study include"
}

pdf_template =\
"""
You are a world class researcher. Based on the following research paper, answer the following:
{prompt}

{context}
"""