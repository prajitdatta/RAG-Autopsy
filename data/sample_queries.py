"""
data/sample_queries.py

Test queries with ground truth answers and source attribution.
Used to evaluate retrieval correctness and generation faithfulness.
"""

QUERIES = [
    {
        "id": "q01",
        "query": "How long do I have to return an item?",
        "ground_truth": "30 days from the purchase date for most items.",
        "source_doc": "refund_policy",
        "key_facts": ["30 days", "original condition", "proof of purchase"],
        "negative_facts": ["60 days", "90 days", "no time limit"],
    },
    {
        "id": "q02",
        "query": "Can I return a digital download?",
        "ground_truth": "No. Digital downloads and software licenses once activated cannot be returned.",
        "source_doc": "refund_policy",
        "key_facts": ["cannot be returned", "digital downloads", "once activated"],
        "negative_facts": ["can be returned", "30 day return", "full refund"],
    },
    {
        "id": "q03",
        "query": "How long does a refund take to process?",
        "ground_truth": "5-7 business days. Credit card refunds may take an additional 3-5 business days to appear.",
        "source_doc": "refund_policy",
        "key_facts": ["5-7 business days", "original payment method", "3-5 business days"],
        "negative_facts": ["24 hours", "instant", "2 weeks"],
    },
    {
        "id": "q04",
        "query": "What rating do I need to get a promotion?",
        "ground_truth": "A rating of 4 or higher in the two most recent consecutive reviews, plus at least 18 months in the current role.",
        "source_doc": "employee_handbook",
        "key_facts": ["rating of 4", "two consecutive reviews", "18 months"],
        "negative_facts": ["rating of 3", "one review", "12 months"],
    },
    {
        "id": "q05",
        "query": "What salary increase do I get for a rating of 5?",
        "ground_truth": "A merit increase of 7-10% plus eligibility for spot bonuses.",
        "source_doc": "employee_handbook",
        "key_facts": ["7-10%", "merit increase", "spot bonuses"],
        "negative_facts": ["4-6%", "2-3%", "no bonus"],
    },
    {
        "id": "q06",
        "query": "What are the core working hours for remote employees?",
        "ground_truth": "10:00 AM to 3:00 PM in their designated time zone.",
        "source_doc": "employee_handbook",
        "key_facts": ["10:00 AM", "3:00 PM", "designated time zone"],
        "negative_facts": ["9 AM to 5 PM", "flexible hours", "no core hours"],
    },
    {
        "id": "q07",
        "query": "What is the starting dose of Metformin?",
        "ground_truth": "500 mg twice daily with meals, titrated to maximum 2000 mg/day.",
        "source_doc": "medical_guidelines",
        "key_facts": ["500 mg", "twice daily", "with meals", "2000 mg/day"],
        "negative_facts": ["1000 mg", "once daily", "before meals"],
    },
    {
        "id": "q08",
        "query": "What is the HbA1c target for most adults with Type 2 diabetes?",
        "ground_truth": "Less than 7.0% for most non-pregnant adults.",
        "source_doc": "medical_guidelines",
        "key_facts": ["less than 7.0%", "non-pregnant adults"],
        "negative_facts": ["less than 8.0%", "less than 6.0%", "6.5%"],
    },
    {
        "id": "q09",
        "query": "How long does the battery last with ANC on?",
        "ground_truth": "28 hours at 50% volume.",
        "source_doc": "product_spec",
        "key_facts": ["28 hours", "50% volume", "ANC on"],
        "negative_facts": ["40 hours", "20 hours", "all conditions"],
    },
    {
        "id": "q10",
        "query": "What was the total revenue in FY2024?",
        "ground_truth": "€847.3 million, a 12.4% increase over FY2023.",
        "source_doc": "financial_report",
        "key_facts": ["€847.3 million", "12.4%", "FY2024"],
        "negative_facts": ["€753.8 million", "FY2023", "15% increase"],
    },
]


def get_query(query_id: str) -> dict:
    """Retrieve a query by ID."""
    for q in QUERIES:
        if q["id"] == query_id:
            return q
    raise KeyError(f"Unknown query: {query_id}")


def get_queries_for_doc(doc_id: str) -> list[dict]:
    """Return all queries associated with a given document."""
    return [q for q in QUERIES if q["source_doc"] == doc_id]
