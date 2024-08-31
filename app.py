import streamlit as st
from config import setup_environment
from rag_initializer import initialize_rag_chain
from chatbot import chatbot_response

setup_environment()

st.set_page_config(page_title="Mini AI Bot for Englishfirm.com", page_icon="🤖", layout="wide")

def main():
    st.title("Mini AI Bot for Englishfirm.com")
    st.sidebar.info("Englishfirm is the one of the best PTE coaching academies in Sydney.  Among 52 PTE institutes in Sydney, Englishfirm is the only training centre in Sydney that offers 100% one-on-one coaching. Englishfirm has 2 branches in Sydney, operating 7 day a week. We operate from Sydney CBD campus (Pitt Street) and Parramatta.")
    st.sidebar.warning("Disclaimer: This chatbot provides information related to Englishfirm's IELTS, PTE, and Spoken English coaching services. While we strive to offer accurate and up-to-date information, this should not be considered a substitute for professional educational advice. For personalized guidance and assistance, please contact our experts directly.")
    rag_chain = initialize_rag_chain()
    
    if rag_chain is None:
        st.error("Failed to initialize the chatbot. Please try again later.")
        return
    
    st.write("Welcome to Englishfirm! I'm here to assist you with any questions regarding our IELTS, PTE, and Spoken English coaching. How can I help you today?")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = chatbot_response(prompt, rag_chain)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    
if __name__ == "__main__":
    main()