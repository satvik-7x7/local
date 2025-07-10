from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# LOADING VIDEO

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

# SPLITTING TRANSCRIPUT INTO CHUNKS 

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# EMBEDDING CHUKENS TO VECTOR STORE 

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.index_to_docstore_id

# RETIVEING 

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# AUGMENTAION 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# PROMPT

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say Insuficient context to give and proper answer

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
# BUILDING CHAIN

###
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke('Can you summarize the video'))