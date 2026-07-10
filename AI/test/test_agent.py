from app.graph.agent import build_agent


agent = build_agent()


# Same thread_id = same conversation memory
config = {
    "configurable": {
        "thread_id": "test-user"
    }
}


# Initial empty state
first_message = True


while True:

    # Get user input
    question = input("\nUSER: ")

    # Exit condition
    if question.lower() == "end":
        print("\nConversation ended.")
        break


    # First message needs initial state
    if first_message:

        state = {
            "question": question,
            "requested_year": None,
            "evidence": [],
            "answer": "",
            "audit_log": [],
            "history": []
        }

        first_message = False

    else:

        # For following messages, LangGraph restores
        # previous state using the thread_id
        state = {
            "question": question,
            "evidence": [],
            "answer": "",
            "audit_log": []
        }


    result = agent.invoke(
        state,
        config=config
    )


    print("\nASSISTANT:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)


    print("\nAUDIT LOG:")
    for item in result["audit_log"]:
        print("-", item)
