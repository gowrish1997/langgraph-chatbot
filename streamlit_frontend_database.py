"""
Streamlit frontend for LangGraph chatbot with persistent conversation database.
Supports multiple conversation threads with streaming responses.
"""

import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Constants ==================
SIDEBAR_TITLE = "LangGraph Chatbot"
SIDEBAR_HEADER = "My Conversations"
CHAT_INPUT_PLACEHOLDER = "Type your message here..."
DEFAULT_THREAD_LABEL = "Chat"

# ================== Utility Functions ==================

def generate_thread_id() -> str:
    """Generate a unique thread ID for a new conversation."""
    return str(uuid.uuid4())


def get_chat_config(thread_id: str) -> dict:
    """Create configuration dict for chatbot with given thread_id."""
    return {"configurable": {"thread_id": thread_id}}


def reset_chat() -> None:
    """Start a new conversation thread and reset chat history."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    logger.info(f"New chat created with thread_id: {thread_id}")


def add_thread(thread_id: str) -> None:
    """Add a thread ID to the chat threads list if not already present."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    """
    Load conversation history for a given thread ID.
    
    Args:
        thread_id: The unique identifier for the conversation thread
        
    Returns:
        List of messages from the conversation, or empty list if none exist
    """
    try:
        config = get_chat_config(thread_id)
        state = chatbot.get_state(config=config)
        messages = state.values.get("messages", [])
        logger.info(f"Loaded {len(messages)} messages from thread {thread_id}")
        return messages
    except Exception as e:
        logger.error(f"Error loading conversation for thread {thread_id}: {str(e)}")
        return []


def format_thread_label(thread_id: str, index: int) -> str:
    """Format a user-friendly label for a thread."""
    truncated_id = str(thread_id)[:8]
    return f"{DEFAULT_THREAD_LABEL} {index + 1} ({truncated_id}...)"


def convert_messages_to_dict(messages: list) -> list:
    """
    Convert LangChain messages to dictionary format for display.
    
    Args:
        messages: List of LangChain message objects
        
    Returns:
        List of dictionaries with 'role' and 'content' keys
    """
    formatted_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted_messages.append({"role": role, "content": msg.content})
    return formatted_messages

# ================== Session State Initialization ==================

def initialize_session_state() -> None:
    """Initialize all required session state variables."""
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []
        
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()
        
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads()
    
    # Ensure current thread is in the threads list
    add_thread(st.session_state["thread_id"])


initialize_session_state()


# ================== Sidebar UI ==================

st.sidebar.title(SIDEBAR_TITLE)

if st.sidebar.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header(SIDEBAR_HEADER)

# Display conversation threads in reverse order (newest first)
for idx, thread_id in enumerate(reversed(st.session_state["chat_threads"])):
    thread_label = format_thread_label(thread_id, len(st.session_state["chat_threads"]) - idx - 1)
    
    if st.sidebar.button(thread_label, key=f"thread_{thread_id}", use_container_width=True):
        try:
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            st.session_state["message_history"] = convert_messages_to_dict(messages)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading conversation: {str(e)}")
            logger.error(f"Error loading thread {thread_id}: {str(e)}")

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear All Chats", key="clear_all_btn"):
    st.warning("Clear all chats feature coming soon!")


# ================== Main Chat UI ==================

# Configure page
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="wide")
st.title("💬 LangGraph Chatbot")

# Display conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input(CHAT_INPUT_PLACEHOLDER)

if user_input:
    # Add user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and stream assistant response
    try:
        config = get_chat_config(st.session_state["thread_id"])
        
        def stream_assistant_response():
            """Stream AI responses from the chatbot."""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                # Yield only assistant token responses
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        # Display and capture streamed response
        with st.chat_message("assistant"):
            ai_response = st.write_stream(stream_assistant_response())
        
        # Store assistant response in history
        if ai_response:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_response}
            )
            logger.info(f"Message processed for thread {st.session_state['thread_id']}")
        else:
            logger.warning("Empty response received from chatbot")
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        st.error(f"❌ Error processing your message: {str(e)}")
        # Remove the user message if response failed
        st.session_state["message_history"].pop()