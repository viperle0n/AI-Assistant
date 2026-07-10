"""Streamlit frontend for the Insurance Policy AI Assistant. Communicates with the FastAPI backend."""


import streamlit as st
import requests
import uuid


API_URL = "http://127.0.0.1:8000/api/ask"


# Page configuration
st.set_page_config(
    page_title="Policy AI Assistant",
    page_icon="📄"
)


st.title("📄 Insurance Policy AI Assistant")

st.write(
    "Ask questions about internal policies, "
    "historical guidelines, and compliance procedures."
)


# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(
        uuid.uuid4()
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# User input
question = st.chat_input(
    "Ask a policy question..."
)



if question:
    # Display user message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):
        st.write(question)


    # Call backend API
    try:
        response = requests.post(

            API_URL,

            json={

                "user_id": "employee001",
                "session_id":
                    st.session_state.session_id,
                "question": question

            },

            timeout=60

        )



        # Check HTTP status
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]

        else:
            answer = (
                "⚠️ The policy assistant returned "
                "an error.\n\n"
                f"Status code: {response.status_code}"
            )



    except requests.exceptions.ConnectionError:
        answer = (
            "⚠️ Cannot connect to the Policy API.\n\n"
            "Please check that the backend server "
            "is running."
        )


    except requests.exceptions.Timeout:
        answer = (
            "⚠️ The request took too long.\n\n"
            "Please retry."
        )


    except Exception as e:
        answer = (
            "⚠️ Unexpected error occurred.\n\n"
            f"{str(e)}"
        )


    # Display response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    with st.chat_message(
        "assistant"
    ):

        st.write(answer)
