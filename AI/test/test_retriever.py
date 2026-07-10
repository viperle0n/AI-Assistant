from app.retrieval.retriever import retrieve_policy_evidence


question = """
According to the 2020 policy,
a customer bought a product 10 days ago.
They have no receipt but show a bank transaction.
Can I approve the refund?
"""


results = retrieve_policy_evidence(question)


for item in results:

    print("\n========================")

    print("SOURCE:", item["metadata"].get("source"))
    print("PAGE:", item["metadata"].get("page"))
    print("YEAR:", item["metadata"].get("year"))

    print("\nTEXT:")
    print(item["content"])
