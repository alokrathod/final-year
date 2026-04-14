from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def generate_phase2_pdf(analysis, architecture, tech_stack, structure, filename="phase2_output.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        spaceAfter=10
    )

    normal_style = styles['Normal']

    code_style = ParagraphStyle(
        name='CodeStyle',
        fontName='Courier',
        fontSize=9,
        leading=12,
        backColor="#F5F5F5",
        leftIndent=10,
        spaceAfter=15
    )

    content = []

    # Title
    content.append(Paragraph("Phase 2: Project Structure Generation", title_style))

    # ---------------- SYSTEM ANALYSIS ----------------
    content.append(Paragraph("1. System Analysis", heading_style))

    content.append(Paragraph(f"<b>System Type:</b> {analysis.get('system_type', '')}", normal_style))
    content.append(Spacer(1, 5))

    content.append(Paragraph("<b>Features:</b>", normal_style))
    for f in analysis.get("features", []):
        content.append(Paragraph(f"• {f}", normal_style))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Non-Functional Requirements:</b>", normal_style))
    for nf in analysis.get("non_functional", []):
        content.append(Paragraph(f"• {nf}", normal_style))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"<b>Complexity:</b> {analysis.get('complexity', '')}", normal_style))
    content.append(Spacer(1, 20))

    # ---------------- ARCHITECTURE ----------------
    content.append(Paragraph("2. Architecture", heading_style))

    content.append(Paragraph(f"<b>Style:</b> {architecture.get('architecture_style', '')}", normal_style))
    content.append(Paragraph(f"<b>Pattern:</b> {architecture.get('design_pattern', '')}", normal_style))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Justification:</b>", normal_style))
    content.append(Paragraph(architecture.get("justification", ""), normal_style))
    content.append(Spacer(1, 20))

    # ---------------- TECH STACK ----------------
    content.append(Paragraph("3. Tech Stack", heading_style))

    content.append(Paragraph(f"<b>Frontend:</b> {tech_stack.get('frontend', '')}", normal_style))
    content.append(Paragraph(f"<b>Backend:</b> {tech_stack.get('backend', '')}", normal_style))
    content.append(Paragraph(f"<b>Database:</b> {tech_stack.get('database', '')}", normal_style))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Tools:</b>", normal_style))
    for tool in tech_stack.get("tools", []):
        content.append(Paragraph(f"• {tool}", normal_style))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Justification:</b>", normal_style))
    content.append(Paragraph(tech_stack.get("justification", ""), normal_style))
    content.append(Spacer(1, 20))

    # ---------------- PROJECT STRUCTURE ----------------
    content.append(Paragraph("4. Project Structure", heading_style))

    content.append(Preformatted(structure, code_style))

    doc.build(content)

    print(f"✅ Clean PDF generated: {filename}")