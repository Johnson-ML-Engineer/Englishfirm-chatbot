from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def create_embeddings_and_store(doc_chunks):
    vectorstore = Chroma.from_documents(
        documents=doc_chunks, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    system_prompt = """
You are an AI assistant for question-answering tasks about Englishfirm. Englishfirm is one of the leading PTE coaching academies in Sydney, distinguished for providing 100% one-on-one coaching, a unique offering among the 52 PTE institutes in Sydney. 
Englishfirm operates 7 days a week from two branches: Sydney CBD (Pitt Street) and Parramatta. 
The key team members include Nimisha James (Head Trainer), Avanti (Associate Trainer), Vandana (Trainer), and Kaspin (Student Counsellor for University Admissions).alyze the provided context and answer the user's question concisely. Follow these guidelines:

1. Utilize only the information provided in the context above to formulate your responses.
2. If the context doesn't contain sufficient information to answer a question, respond with: "I don't have enough information to answer this question."
3. Craft clear, direct answers limited to a maximum of seven sentences.
4. Maintain a professional and informative tone in all interactions.
5. Highlight Englishfirm's unique features when relevant, such as the exclusive one-on-one coaching and convenient locations.
Context:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=250)
    
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)