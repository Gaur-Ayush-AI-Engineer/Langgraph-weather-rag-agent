import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()


class RAGTool:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, chunker=None,
                 rerank_threshold: float = 0.0, query_strategy=None):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.client = QdrantClient(url="http://localhost:6333")
        self.vectorstore = None
        self.current_pdf_name = None
        self.current_collection_name = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = chunker
        self.rerank_threshold = rerank_threshold
        self.query_strategy = query_strategy  # QueryRewriter | MultiQueryRetriever | None
    
    def _sanitize_collection_name(self, pdf_name: str) -> str:
        """
        Create a valid Qdrant collection name from PDF filename
        
        Args:
            pdf_name: Original PDF filename
            
        Returns:
            Sanitized collection name
        """
        # Remove .pdf extension
        name = pdf_name.replace('.pdf', '')
        # Replace spaces and special chars with underscore
        name = ''.join(c if c.isalnum() else '_' for c in name)
        # Include chunker type + chunk size so each config gets its own collection
        chunker_tag = "struct" if self.chunker is not None else f"c{self.chunk_size}"
        name = f"pdf_{name[:45]}_{chunker_tag}"
        return name.lower()
    
    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load and process PDF into its own vector store collection
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_name = os.path.basename(pdf_path)
            collection_name = self._sanitize_collection_name(pdf_name)
            
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                print(f"📦 Creating new collection: {collection_name}")
                
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Split into chunks — use custom chunker if provided
                if self.chunker is not None:
                    splits = self.chunker.split_documents(documents)
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len
                    )
                    splits = text_splitter.split_documents(documents)
                
                # Store in Qdrant
                QdrantVectorStore.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    url="http://localhost:6333",
                    collection_name=collection_name
                )
                
                print(f"✅ Loaded {len(splits)} chunks into collection: {collection_name}")
            else:
                print(f"📚 Collection already exists: {collection_name}")
            
            # Switch to this collection
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings
            )
            
            self.current_pdf_name = pdf_name
            self.current_collection_name = collection_name
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading PDF: {str(e)}")
            return False
    
    def switch_to_pdf(self, pdf_name: str) -> bool:
        """
        Switch to an existing PDF's collection
        
        Args:
            pdf_name: Name of the PDF file
            
        Returns:
            True if collection exists and switched, False otherwise
        """
        collection_name = self._sanitize_collection_name(pdf_name)
        
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings
            )
            self.current_pdf_name = pdf_name
            self.current_collection_name = collection_name
            print(f"🔄 Switched to collection: {collection_name}")
            return True
        else:
            print(f"⚠️ Collection not found: {collection_name}")
            return False
    
    def _rerank_documents(self, query: str, documents: List) -> List:
        """
        Rerank documents using Flashrank
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Reranked documents (top 3)
        """
        from flashrank import Ranker, RerankRequest
        
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        print("\n--- Reranking Scores ---")
        for result in results[:5]:
            print(f"Doc {result['id']}: Score = {result['score']:.4f}")

        # Apply threshold — keep top 3 that meet the minimum score
        above_threshold = [r for r in results[:3] if r['score'] >= self.rerank_threshold]

        # Fallback: if all chunks are below threshold, keep the single best one
        if not above_threshold:
            print(f"⚠️  All scores below threshold ({self.rerank_threshold}), using top-1 as fallback")
            above_threshold = [results[0]]

        top_indices = [r['id'] for r in above_threshold]
        return [documents[i] for i in top_indices]
    
    def query(self, question: str, chat_history: List = None) -> Dict:
        """
        Answer question using RAG with reranking
        
        Args:
            question: User question
            chat_history: List of previous messages
            
        Returns:
            Dict with answer and sources
        """
        if not self.vectorstore:
            return {
                "answer": "No PDF loaded. Please upload a PDF first.",
                "sources": [],
                "contexts": []
            }
        
        if chat_history is None:
            chat_history = []
        
        try:
            from query_strategies import QueryRewriter, MultiQueryRetriever

            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

            # Apply query strategy before retrieval
            if isinstance(self.query_strategy, QueryRewriter):
                search_query = self.query_strategy.rewrite(question)
                docs = retriever.invoke(search_query)
            elif isinstance(self.query_strategy, MultiQueryRetriever):
                docs = self.query_strategy.retrieve_and_deduplicate(question, retriever, k=5)
            else:
                print(f"\n   Query: {question}")
                docs = retriever.invoke(question)

            print(f"\n--- Retrieved {len(docs)} documents from {self.current_collection_name} ---")
            
            if not docs:
                return {
                    "answer": "No relevant information found in the document.",
                    "sources": [],
                    "contexts": []
                }
            
            # Rerank documents
            reranked_docs = self._rerank_documents(question, docs)
            
            # Create context
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            
            # Create QA prompt
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so."),
                ("system", "Context: {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Generate answer
            chain = qa_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "input": question,
                "context": context,
                "chat_history": chat_history
            })
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in reranked_docs
                ],
                "contexts": [doc.page_content for doc in reranked_docs]
            }
            
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "contexts": []
            }


# Test
if __name__ == "__main__":
    rag = RAGTool()
    
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        rag.load_pdf(pdf_path)
        result = rag.query("What is this document about?")
        print(f"\nAnswer: {result['answer']}")
    else:
        print(f"PDF not found: {pdf_path}")