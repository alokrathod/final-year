from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


def _is_ai_suggested(line, ai_suggested_reqs):
    """
    Check if a line matches any of the AI-suggested requirement texts.
    Matching is done by substring to handle minor formatting differences.
    """
    if not ai_suggested_reqs:
        return False
    line_lower = line.lower().strip()
    for req in ai_suggested_reqs:
        # Match if the core of the req text appears in the line
        req_core = req.lower().strip()[:60]
        if req_core and req_core in line_lower:
            return True
    return False


def export_srs_to_pdf(srs_text, filename="final_srs.pdf", ai_suggested_reqs=None):
    """
    Exports the SRS to a formatted PDF.

    Args:
        srs_text (str): Full SRS text
        filename (str): Output PDF filename
        ai_suggested_reqs (list of str): Requirement texts to render as
                                         Bold + [AI-SUGGESTED] label
    """

    if ai_suggested_reqs is None:
        ai_suggested_reqs = []

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()

    # ── Styles ──────────────────────────────────────────────

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=14
    )

    h2_style = ParagraphStyle(
        'Heading2Style',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=6
    )

    h3_style = ParagraphStyle(
        'Heading3Style',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4
    )

    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceAfter=4
    )

    # Style for AI-suggested requirements: bold text
    ai_suggested_style = ParagraphStyle(
        'AISuggestedStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceAfter=4,
        textColor=colors.black,
    )

    # ── Render lines ─────────────────────────────────────────

    lines = srs_text.split("\n")

    for line in lines:

        line_stripped = line.strip()

        if not line_stripped:
            elements.append(Spacer(1, 0.2 * inch))
            continue

        # Title
        if line_stripped.startswith("# "):
            text = line_stripped.replace("# ", "")
            elements.append(Paragraph(text, title_style))

        # Section heading
        elif line_stripped.startswith("## "):
            text = line_stripped.replace("## ", "")
            elements.append(Paragraph(text, h2_style))

        # Subsection heading
        elif line_stripped.startswith("### "):
            text = line_stripped.replace("### ", "")
            elements.append(Paragraph(text, h3_style))

        else:
            # Escape XML special characters
            safe_line = (
                line_stripped
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # Check if this line is an AI-suggested requirement
            if _is_ai_suggested(line_stripped, ai_suggested_reqs):
                # Render bold with [AI-SUGGESTED] label in a distinct color
                ai_text = (
                    f'<b>{safe_line}</b>'
                    f'&nbsp;&nbsp;<font color="#B8860B"><b>[AI-SUGGESTED]</b></font>'
                )
                elements.append(Paragraph(ai_text, ai_suggested_style))
            else:
                elements.append(Paragraph(safe_line, normal_style))

    doc.build(elements)
    print(f"\nSRS PDF saved as: {filename}")


def export_srs_to_pdf_with_legend(srs_text, filename="final_srs.pdf", ai_suggested_reqs=None):
    """
    Same as export_srs_to_pdf but prepends a legend box explaining
    the [AI-SUGGESTED] label to the reader.
    """

    if ai_suggested_reqs is None:
        ai_suggested_reqs = []

    # Prepend legend only if there are AI suggestions
    if ai_suggested_reqs:
        legend = (
            "\n⚠ NOTE: Requirements marked [AI-SUGGESTED] were not part of the original "
            "user input. They were identified by the AI as potentially helpful or implied "
            "by the domain. Please review and decide whether to keep them.\n"
        )
        srs_text = legend + "\n" + srs_text

    export_srs_to_pdf(srs_text, filename, ai_suggested_reqs)