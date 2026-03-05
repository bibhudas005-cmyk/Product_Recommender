from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from flipkart.config import config 
from flipkart.data_ingestion import DataIngestor 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory #used for adding message history into the rag pipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory  #the last two imports are used for forming the message history


class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model=ChatGroq(model=config.RAG_MODEL,temperature=0.5)
        self.history_store={} #Empty dict which stores all the conversation history for different sesssion

 
    def _get_history(self,session_id:str)->BaseChatMessageHistory: 
        if session_id not in self.history_store:
            self.history_store[session_id]=ChatMessageHistory() #This chat Message history is like a data type which is derived from BaseChatMessageHistory
            return self.history_store[session_id]

    def build_chain(self):
        retriever=self.vector_store.as_retriever(search_kwargs={"k":3}) #retrieve 3 relevant info based on the requirement of search

        #Rewrite the question based on the chat history
        context_prompt=ChatPromptTemplate.from_messages([
            ("system","Given the chat history and user question, rewrite it as standalone question"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ])


        #Tell the llm to behave in such a way that it uses the context to answer the question
        qa_prompt=ChatPromptTemplate.from_messages([
            ("system","""You're an ecommerce bot answering post-related queries using reviews and titles.
                      Stick to the context. Be concise and helpful.\n\nCONTEXT:{context}\n\n QUESTION:{input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")])


        history_aware_retriever=create_history_aware_retriever(
            self.model,retriever,context_prompt
        )

        question_answer_chain=create_stuff_documents_chain(
            self.model,qa_prompt
        )


        rag_chain=create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_message_key="chat_history",
            output_message_key="answer",


        )