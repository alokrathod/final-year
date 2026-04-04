from modules.specificity_q1 import compute_specificity_Q1


def main():

    # 🔹 Hardcoded sample requirements
    requirements = [
    # ── CLEAR (should be UNAMBIGUOUS) ──
    "REQ-001: The system shall allow users to register using email and password.",
    "REQ-002: The system shall return search results within 2 seconds.",
    "REQ-003: The system shall allow users to upload files up to 10MB.",
    "REQ-004: The system shall send an email notification when a password is reset.",
    "REQ-005: The system shall allow users to borrow up to 3 books at a time.",

    # ── AMBIGUOUS (should be detected) ──
    "REQ-006: The system shall provide fast response time.",
    "REQ-007: The system shall be user-friendly.",
    "REQ-008: The system shall allow users to view reports and export them.",
    "REQ-009: The system shall support login using email or phone.",
    "REQ-010: The system shall send notifications to users."
    ]

    print("\n=== TESTING Q1 SPECIFICITY WITH SAMPLE REQUIREMENTS ===\n")

    # 🔹 Run Q1
    score = compute_specificity_Q1(requirements)

    print(f"\nFinal Q1 Specificity (Likert Scale): {round(score * 5, 2)}")


if __name__ == "__main__":
    main()