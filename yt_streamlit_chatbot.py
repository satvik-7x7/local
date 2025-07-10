import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Transcript Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        padding: 1rem 0;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #191970;
    }
    .video-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return ID if already provided"""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        # Regular expression to extract the video ID
        regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(regex, url_or_id)
        if match:
            return match.group(1)
        else:
            return None
    else:
        # Assume it's already an ID
        return url_or_id.strip()


def load_transcript(video_id):
    """Load transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, "No captions available for this video."
    except Exception as e:
        return None, f"Error loading transcript: {str(e)}"

def process_transcript(transcript):
    """Split transcript into chunks and create vector store"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def create_rag_chain(vector_store):
    """Create the RAG chain"""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say "Insufficient context to give a proper answer"

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    
    return main_chain

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'transcript_loaded' not in st.session_state:
    st.session_state.transcript_loaded = False

# Main UI
st.markdown("<h1 class='main-header'>🎥 YouTube Transcript Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Ask questions about any YouTube video with captions!</p>", unsafe_allow_html=True)

# Sidebar for video input
with st.sidebar:
    st.header("📹 Video Setup")
    
    # Video input
    video_input = st.text_input(
        "Enter YouTube URL or Video ID:",
        placeholder="e.g., https://youtube.com/watch?v=... or Gfr50f6ZBvo",
        help="Paste a YouTube URL or just the video ID"
    )
    
    if st.button("Load Video Transcript", type="primary"):
        if video_input:
            video_id = extract_video_id(video_input)
            
            if video_id:
                with st.spinner("Loading transcript..."):
                    transcript, error = load_transcript(video_id)
                    
                    if transcript:
                        with st.spinner("Processing transcript..."):
                            try:
                                st.session_state.vector_store = process_transcript(transcript)
                                st.session_state.chain = create_rag_chain(st.session_state.vector_store)
                                st.session_state.current_video_id = video_id
                                st.session_state.transcript_loaded = True
                                st.session_state.chat_history = []  # Clear previous chat
                                st.success("✅ Transcript loaded successfully!")
                            except Exception as e:
                                st.error(f"Error processing transcript: {str(e)}")
                    else:
                        st.error(f"❌ {error}")
            else:
                st.error("❌ Invalid YouTube URL or Video ID")
        else:
            st.error("❌ Please enter a YouTube URL or Video ID")
    
    # Video info
    if st.session_state.current_video_id:
        st.markdown(f"""
        <div class="video-info" style="background-color: #FF6B6B; color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4>🎬 Current Video</h4>
            <p><strong>Video ID:</strong> {st.session_state.current_video_id}</p>
            <p><strong>Status:</strong> {'✅ Ready' if st.session_state.transcript_loaded else '❌ Not loaded'}</p>
            <a href="https://youtube.com/watch?v={st.session_state.current_video_id}" target="_blank" style="color: #87CEFA;">
                🔗 Watch on YouTube
            </a>
        </div>
        """, unsafe_allow_html=True)

    
    # Instructions
    st.markdown("""
    ### 📝 Instructions
    1. **Enter** a YouTube URL or Video ID
    2. **Click** "Load Video Transcript"
    3. **Ask** questions about the video content
    4. **Get** answers based on the transcript
    
    ### 💡 Sample Questions
    - "Can you summarize the video?"
    - "What are the main points discussed?"
    - "What does the speaker say about [topic]?"
    """)

# Main content area
if st.session_state.transcript_loaded:
    st.header("💬 Ask Questions")
    
    # Chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Your question:",
            placeholder="Ask anything about the video...",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary")
    
    # Handle question submission
    if ask_button and question:
        if st.session_state.chain:
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("📋 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}{'...' if len(chat['question']) > 50 else ''}", expanded=(i == 0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.markdown("---")
    
    # Quick actions
    st.header("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Summarize Video"):
            if st.session_state.chain:
                with st.spinner("Generating summary..."):
                    try:
                        summary = st.session_state.chain.invoke("Can you provide a comprehensive summary of this video?")
                        st.session_state.chat_history.append({"question": "Summarize the video", "answer": summary})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
    
    with col2:
        if st.button("🔍 Key Points"):
            if st.session_state.chain:
                with st.spinner("Extracting key points..."):
                    try:
                        key_points = st.session_state.chain.invoke("What are the main key points or takeaways from this video?")
                        st.session_state.chat_history.append({"question": "What are the key points?", "answer": key_points})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting key points: {str(e)}")
    
    with col3:
        if st.button("🗂️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #FF6B6B; border-radius: 10px; margin: 2rem 0;">
        <h2>👋 Welcome to YouTube Transcript Q&A!</h2>
        <p style="font-size: 1.1rem;">
            Get started by entering a YouTube URL or Video ID in the sidebar.
        </p>
        <p>
            This tool will analyze the video transcript and answer your questions about the content.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("""
    ### ✨ Features
    
    - **🎯 Accurate Answers**: Get precise answers based on video transcripts
    - **📊 Smart Retrieval**: Uses advanced RAG to find relevant content
    - **💬 Chat Interface**: Interactive Q&A with chat history
    - **🚀 Quick Actions**: One-click summaries and key points
    - **📱 Responsive**: Works on desktop and mobile devices
    """)

# Footer
st.markdown("""
---
<p style="text-align: center; color: #666; font-size: 0.8rem;">
    Made with ❤️ using Streamlit | Powered by Google Gemini & LangChain
</p>
""", unsafe_allow_html=True)