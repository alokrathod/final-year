from modules.llm import call_llm


def generate_srs(requirements, rag_context=None):

    formatted_reqs = "\n".join(
        [f"{req['id']}: {req['text']}" for req in requirements]
    )

    # RAG is used ONLY to learn writing style and section formatting.
    # A separate LLM call extracts only the structural patterns,
    # completely stripping all domain content before it reaches
    # the generator — works dynamically for any input domain.
    format_style = ""
    if rag_context:
        style_prompt = f"""
You are a document structure analyst.

Read the SRS excerpt below and extract ONLY:
- The writing style used for requirements (e.g. "The system shall...")
- The section heading format and hierarchy
- The level of detail used in each section
- Any formatting conventions (numbering, indentation, bullet style)

DO NOT extract or include:
- Any system names, product names, or organization names
- Any domain-specific content, features, or functionality
- Any numbers, metrics, or scale information
- Anything describing what the original system does

Output a short style guide (max 10 lines) describing only the
structural and formatting patterns observed.

SRS Excerpt:
{rag_context}
"""
        format_style = call_llm(style_prompt, temperature=0.1)

    rag_section = f"""
Writing Style & Format Guide (extracted from reference SRS documents):
{format_style}

IMPORTANT: The above is ONLY a formatting guide.
All content must come exclusively from the Requirements section below.
""" if format_style else ""

    prompt = f"""
Generate a complete Software Requirements Specification (SRS) following IEEE 830 standard.

=== STRICT CONTENT RULES ===
1. Use ONLY the requirements listed below as content.
2. Do NOT invent, infer, or add any features not in the requirements.
3. Do NOT reference any external systems, tools, or products.
4. Do NOT use placeholder text like "[No specific ... provided]".
   For sections with no direct requirement, write one brief generic sentence.

=== REQUIRED IEEE 830 STRUCTURE ===
1. Introduction
   1.1 Purpose
   1.2 Scope
   1.3 Definitions, Acronyms, and Abbreviations
   1.4 References
   1.5 Overview

2. Overall Description
   2.1 Product Perspective
   2.2 Product Functions
   2.3 User Characteristics
   2.4 Constraints
   2.5 Assumptions and Dependencies

3. Functional Requirements
   - Each requirement MUST be atomic
   - Format: REQ-XXX: The system shall ...
   - One requirement per line

4. Non-Functional Requirements
   4.1 Performance Requirements
   4.2 Security Requirements
   4.3 Usability Requirements
   4.4 Reliability Requirements
   4.5 Maintainability Requirements
   4.6 Portability Requirements

5. External Interface Requirements
   5.1 User Interfaces
   5.2 Hardware Interfaces
   5.3 Software Interfaces
   5.4 Communication Interfaces

6. System Constraints

7. Assumptions and Dependencies

=== REQUIREMENTS ===
{formatted_reqs}

{rag_section}

Return the FULL SRS.
"""

    return call_llm(prompt)