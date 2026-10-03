SCHEMES = [
    {"name": "PM-KISAN",
     "benefit": "₹6,000/year in 3 installments of ₹2,000 to eligible landholding farmer families.",
     "eligibility": "All landholding farmer families except institutional landholders, income-tax payers, and government employees.",
     "apply": "pmkisan.gov.in or nearest CSC. Helpline 155261."},
    {"name": "PM Fasal Bima Yojana",
     "benefit": "Crop insurance against natural calamity, pests, and disease.",
     "eligibility": "All farmers growing notified crops in notified areas.",
     "apply": "pmfby.gov.in, bank, or CSC before the state cut-off date."},
    {"name": "Kisan Credit Card",
     "benefit": "Short-term credit up to ₹3 lakh at 7% (effectively 4% with prompt repayment).",
     "eligibility": "All farmers — owner, tenant, oral lessee, sharecropper.",
     "apply": "Any scheduled commercial bank, RRB, or cooperative bank."},
    {"name": "Soil Health Card",
     "benefit": "Free soil testing and nutrient recommendation every 2 years.",
     "eligibility": "All farmers.",
     "apply": "soilhealth.dac.gov.in or state agriculture department."},
]

IPM_RULES = """
For any pest/disease query:
1. First line: cultural/mechanical control (crop rotation, trap crops, neem oil, pheromone traps).
2. Second line: biological control (Trichoderma, Bacillus, NPV).
3. Third line only if farmer insists and severity is high: name the class of chemical (e.g. "a contact insecticide"), never a brand or dose. Say a named agricultural officer must confirm the specific product and dose.
4. Always end with the KVK helpline: 1800-180-1551.
"""