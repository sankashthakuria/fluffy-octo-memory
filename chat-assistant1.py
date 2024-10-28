import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from colorama import Fore, Style, init

# Initialize colorama for colored console output
init(autoreset=True)


class DocumentChatAssistant:
    def __init__(self, docs_dir="documents", db_dir="db"):
        self.docs_dir = Path(docs_dir)
        self.db_dir = db_dir

        # Load embeddings for text similarity
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Split documents into smaller parts for better results
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Each chunk has up to 1000 characters
            chunk_overlap=100,  # Overlapping text to keep context
        )

        # Setup a simple LLM model for answering questions
        self.llm = Ollama(model="llama3.2", temperature=0.5)

        # Check if the vector store (database) exists, and delete if it does
        if os.path.exists(db_dir):
            import shutil

            shutil.rmtree(db_dir)

        # Create a vector store from documents in the specified directory
        self.vectorstore = self._create_vectorstore()

        # Use RetrievalQA instead of ConversationalRetrievalChain to avoid chat history issues
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True,
        )

    def _create_vectorstore(self):
        """Loads and processes documents to create a vector store."""
        print("Loading documents from folder...")
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Processed into {len(chunks)} text chunks")

        # Create and save vector store with embeddings
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings, persist_directory=self.db_dir
        )
        return vectorstore

    def chat(self, query):
        """Handles chat by passing questions to the QA chain."""
        try:
            response = self.qa_chain.invoke(
                {"query": query}
            )  # Use invoke method to avoid deprecation warning
            answer = response["result"]
            sources = [doc.metadata["source"] for doc in response["source_documents"]]
            return {"answer": answer, "sources": sources}
        except Exception as e:
            print(f"Error during chat: {str(e)}")
            return {
                "answer": "There was an error processing your question.",
                "sources": [],
            }


if __name__ == "__main__":
    # Initialize the assistant with default directories
    assistant = DocumentChatAssistant()

    print("Ask a question about the documents! (type 'quit' to exit)")
    while True:
        query = input(f"\n{Fore.BLUE}You: {Style.RESET_ALL}")
        if query.lower() == "quit":
            break

        # Get response from the assistant
        response = assistant.chat(query)
        print(f"\n{Fore.GREEN}Bot: {response['answer']}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Sources:{Style.RESET_ALL} {', '.join(response['sources'])}"
        )
        print("-" * 50)
