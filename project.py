from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import sys

load_dotenv()  # load API keys from .env file


# Fetch transcript from YouTube
def get_yt_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([chunk['text'] for chunk in transcript_list])
        return transcript
    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Error: {str(e)}"


# Join retrieved documents into context text
def format_docs(retrieved_docs):
    context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs)
    return context_text


def main():
    try:
        # --- Step 1: Get YouTube video ID ---
        video_id = input("Enter YouTube video ID: ").strip()
        transcript = get_yt_transcript(video_id)

        if transcript.startswith("Error:") or transcript == "No captions available for this video.":
            print(transcript)
            sys.exit()

        # --- Step 2: Split transcript into chunks ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # --- Step 3: Create embeddings and FAISS index ---
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Cache FAISS index (optional, for performance)
        faiss_path = f"{video_id}_index"
        if os.path.exists(faiss_path):
            vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded saved FAISS index.")
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(faiss_path)
            print("✅ Created and saved new FAISS index.")

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

        # --- Step 4: Define prompt template ---
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Use ONLY the provided transcript context to answer the question.
If the context doesn’t include enough information, say "I'm not sure about that."

Transcript context:
{context}

Question:
{question}
""",
            input_variables=['context', 'question']
        )

        # --- Step 5: Setup the RAG chain ---
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        main_chain = parallel_chain | prompt | model | parser

        # --- Step 6: Interactive Q&A ---
        print("\n✅ Transcript ready. You can now ask questions about the video.")
        while True:
            question = input("\nAsk (or 'quit' to exit): ").strip()
            if question.lower() in ["quit", "exit"]:
                break
            answer = main_chain.invoke(question)
            print("\n🤖 Answer:", answer)

    except Exception as e:
        print("Unexpected error:", str(e))


if __name__ == "__main__":
    main()
