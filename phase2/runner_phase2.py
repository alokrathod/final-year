from phase2.system_analyzer import analyze_system
from phase2.architecture_selector import select_architecture
from phase2.tech_stack_selector import select_tech_stack
from phase2.structure_generator import generate_structure
from phase2.pdf_generator import generate_phase2_pdf

def load_srs(file_path="final_srs.txt"):
    with open(file_path, "r") as f:
        return f.read()


def main():

    print("\n===== PHASE 2: PROJECT STRUCTURE GENERATION =====\n")

    # Load SRS from Phase 1
    srs_text = load_srs()

    # Step 1: Analyze system
    analysis = analyze_system(srs_text)

    # Step 2: Architecture
    architecture = select_architecture(analysis)

    # Step 3: Tech stack
    tech_stack = select_tech_stack(analysis, architecture)

    #Step 4: Structure
    structure = generate_structure(architecture, tech_stack, analysis)

    print("\n===== SYSTEM ANALYSIS =====")
    print(analysis)

    print("\n===== ARCHITECTURE =====")
    print(architecture)

    print("\n===== TECH STACK =====")
    print(tech_stack)

    print("\n===== PROJECT STRUCTURE =====")
    print(structure)

    # Generate PDF
    generate_phase2_pdf(analysis, architecture, tech_stack, structure)


if __name__ == "__main__":
    main()