import streamlit as st
from rag_chain import stream_qwen

# Set page config
st.set_page_config(
    page_title="Fortune Cloud AI Assistant",
    page_icon="☁️",
    layout="centered"
)

# Header
st.title("☁️ Fortune Cloud Assistant")
st.markdown("Ask me anything about courses, placements, schedules, or our offices!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Format chat history for the prompt
def format_chat_history(messages):
    formatted = []
    # Only keep the last 4 messages to save context window and avoid slowing down the local LLM
    recent_messages = messages[-4:]
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What courses do you offer?"):
    
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Format the history to pass to the model
    chat_history_str = format_chat_history(st.session_state.messages[:-1]) # exclude the current prompt
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # We wrap the streaming call in a spinner just in case TTFT (Time To First Token) is slow
        with st.spinner("Thinking..."):
            # Stream the response from our RAG chain
            for chunk in stream_qwen(prompt, chat_history_str):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                
        # Final update without the cursor
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
