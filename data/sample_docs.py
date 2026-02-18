"""
data/sample_docs.py

Synthetic document corpus used across all failure tests.
No external files needed — everything is self-contained.
Each document is realistic enough to expose real failure modes.
"""

# ── Document corpus ───────────────────────────────────────────────────────────

DOCS = {
    "refund_policy": """
Refund and Return Policy — Effective January 1, 2025

Section 1: General Return Window
Customers may return most items within 30 days of the purchase date for a full refund.
Items must be in their original condition, unused, and in original packaging.
Proof of purchase is required for all returns.

Section 2: Non-Returnable Items
The following items cannot be returned under any circumstances:
- Perishable goods including food, flowers, and plants
- Digital downloads and software licenses once activated
- Personalized or custom-made items
- Hazardous materials and flammable liquids
- Intimate apparel and swimwear for hygiene reasons

Section 3: Refund Processing
Approved refunds are processed within 5-7 business days.
Refunds are issued to the original payment method only.
Credit card refunds may take an additional 3-5 business days to appear on statements.
Cash payments are refunded via company check mailed within 10 business days.

Section 4: Defective Items
Defective items may be returned within 90 days of purchase regardless of standard policy.
Customer must provide photographic evidence of the defect before a return label is issued.
Defective items will be replaced first; refund issued only if replacement is unavailable.

Section 5: International Returns
International customers are responsible for all return shipping costs.
Customs duties and taxes paid are non-refundable.
International refunds are processed in USD at the exchange rate on the day of refund.
""",

    "employee_handbook": """
Employee Handbook — Engineering Division, Version 3.2

Chapter 4: Performance Reviews

4.1 Review Cycle
Performance reviews are conducted twice per year: in June and December.
Mid-year reviews focus on goal progress and development needs.
Year-end reviews determine compensation adjustments and promotion eligibility.

4.2 Rating Scale
Employees are rated on a five-point scale:
1 - Needs Improvement: Performance is below expectations in most areas.
2 - Developing: Performance meets some but not all expectations.
3 - Meets Expectations: Solid performance that fully meets role requirements.
4 - Exceeds Expectations: Consistently delivers above what is required.
5 - Outstanding: Exceptional performance that significantly exceeds all expectations.

4.3 Promotion Criteria
To be eligible for promotion, an employee must:
- Receive a rating of 4 or higher in the most recent two consecutive reviews
- Have been in their current role for a minimum of 18 months
- Demonstrate competency in at least 80% of the skills required for the next level
- Receive sponsorship from their direct manager and skip-level manager

4.4 Compensation
Employees rated 3 receive a cost-of-living adjustment of 2-3%.
Employees rated 4 receive a merit increase of 4-6%.
Employees rated 5 receive a merit increase of 7-10% plus eligibility for spot bonuses.
Employees rated 1 or 2 are not eligible for salary increases.

Chapter 5: Remote Work Policy

5.1 Eligibility
All full-time employees who have completed their 90-day probationary period are eligible
for remote work arrangements subject to manager approval.

5.2 Core Hours
All employees must be available and responsive during core hours: 10:00 AM to 3:00 PM
in their designated time zone. Meetings may be scheduled during these hours without
prior approval of the attendee's schedule.

5.3 Equipment
The company provides a laptop and one external monitor for remote employees.
Employees are responsible for their own internet connection and its reliability.
A monthly stipend of €50 is provided for internet and home office expenses.
""",

    "medical_guidelines": """
Clinical Protocol: Type 2 Diabetes Management
Version 2.1 — Internal Use Only

1. Diagnosis Criteria
Type 2 diabetes is diagnosed when:
- Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L) on two separate occasions
- 2-hour plasma glucose ≥ 200 mg/dL during an OGTT
- HbA1c ≥ 6.5% confirmed by repeat testing
- Random plasma glucose ≥ 200 mg/dL with classic symptoms

2. First-Line Treatment
Metformin is the preferred initial pharmacological agent unless contraindicated.
Contraindications include eGFR < 30 mL/min/1.73m², active liver disease, and
known hypersensitivity. Starting dose is 500 mg twice daily with meals, titrated
to a maximum of 2000 mg/day over 4-8 weeks based on tolerability.

3. HbA1c Targets
General target: < 7.0% for most non-pregnant adults.
Less stringent target (< 8.0%): elderly patients, limited life expectancy,
history of severe hypoglycemia, advanced microvascular complications.
More stringent target (< 6.5%): short disease duration, long life expectancy,
no significant cardiovascular disease, if achievable without hypoglycemia.

4. Monitoring
Self-monitoring of blood glucose: at least twice daily for insulin users.
HbA1c: every 3 months until stable, then every 6 months.
Renal function (eGFR, urine albumin): annually.
Lipid panel: annually.
Dilated eye exam: annually.
Foot examination: at every visit.

5. Escalation
Add a second agent if HbA1c target not achieved after 3 months of maximum
tolerated metformin dose. Agent selection based on: presence of cardiovascular
disease (prefer GLP-1 RA or SGLT2i), heart failure (prefer SGLT2i), CKD
(prefer SGLT2i), weight loss needed (prefer GLP-1 RA), hypoglycemia risk
(prefer DPP-4i or GLP-1 RA), cost constraints (prefer SU or TZD).
""",

    "product_spec": """
Product Specification: Nexus Pro X1 Wireless Headphones
Document ID: SPEC-2024-HPX1-v4

Audio Specifications
Driver Size: 40mm dynamic driver with beryllium-coated diaphragm
Frequency Response: 20 Hz - 40 kHz (extended range mode)
Impedance: 32 Ohms ± 15%
Sensitivity: 105 dB SPL / 1mW at 1kHz
Total Harmonic Distortion: < 0.05% at 1kHz, 100dB SPL
Maximum Input Power: 30mW

Connectivity
Bluetooth Version: 5.3 with multipoint connection (up to 2 devices simultaneously)
Supported Codecs: SBC, AAC, aptX, aptX HD, aptX Adaptive, LDAC
Wireless Range: Up to 30 metres in open space (Class 1 Bluetooth)
Latency: 40ms in low-latency mode (aptX Adaptive)
Wired Connection: 3.5mm stereo jack, 1.2m detachable cable included

Battery
Battery Capacity: 800 mAh lithium-ion
ANC On Playtime: 28 hours at 50% volume
ANC Off Playtime: 40 hours at 50% volume
Charging Time: 2 hours via USB-C (18W fast charge supported)
Quick Charge: 10 minutes charging provides 3 hours playback
Standby Time: 200 hours

Active Noise Cancellation
ANC Type: Hybrid feedforward/feedback with 6 microphones
Noise Attenuation: Up to 40 dB in the 100-1000 Hz range
Modes: Maximum ANC, Adaptive ANC, Transparency, Off
Transparency Mode: Natural sound with voice enhancement
Wind Noise Reduction: Integrated in Transparency mode

Physical
Weight: 254g (without cable)
Folding: Yes — ear cups rotate flat, headband folds for travel
Ear Cushion Material: Memory foam with protein leather cover
IP Rating: IPX4 (splash resistant)
Colours: Midnight Black, Arctic White, Deep Navy, Sage Green
""",

    "financial_report": """
Annual Financial Report — FY2024
Meridian Capital Group

Executive Summary
Total revenue for FY2024 was €847.3 million, representing a 12.4% increase
over FY2023 revenue of €753.8 million. Operating income improved to €143.6 million
(16.9% operating margin) compared to €118.2 million (15.7% margin) in FY2023.
Net income attributable to shareholders was €98.4 million, up from €79.1 million.
Basic earnings per share: €2.47 (FY2023: €1.98). Diluted EPS: €2.41 (FY2023: €1.94).

Revenue Breakdown by Segment
Asset Management: €412.1M (48.6% of total) — up 18.3% YoY
Wealth Management: €234.7M (27.7% of total) — up 6.1% YoY
Investment Banking: €156.4M (18.5% of total) — up 9.8% YoY
Other / Corporate: €44.1M (5.2% of total) — up 2.3% YoY

Balance Sheet Highlights (as of December 31, 2024)
Total assets: €12.4 billion
Total liabilities: €9.8 billion
Total equity: €2.6 billion
Tier 1 capital ratio: 14.8% (regulatory minimum: 8.0%)
Leverage ratio: 5.2% (regulatory minimum: 3.0%)
Liquidity coverage ratio: 142% (regulatory minimum: 100%)

Dividend
The Board proposes a final dividend of €0.72 per share for FY2024,
bringing the full-year dividend to €1.20 per share (FY2023: €0.95 per share).
Record date: March 15, 2025. Payment date: April 2, 2025.
"""
}


# ── Flat list of document chunks (pre-split for retrieval tests) ──────────────

def get_all_docs() -> list[dict]:
    """Return all documents as a flat list with metadata."""
    return [
        {"id": doc_id, "content": content.strip(), "source": doc_id}
        for doc_id, content in DOCS.items()
    ]


def get_doc(doc_id: str) -> str:
    """Return a single document by ID."""
    if doc_id not in DOCS:
        raise KeyError(f"Unknown document: {doc_id}. Available: {list(DOCS.keys())}")
    return DOCS[doc_id].strip()


def get_all_text() -> str:
    """Return all documents concatenated."""
    return "\n\n---\n\n".join(DOCS.values())
