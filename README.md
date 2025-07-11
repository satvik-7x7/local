🎥 YouTube Transcript Q&A

A Streamlit-powered web app that lets you interactively query and summarize YouTube videos using their transcripts. It leverages LangChain, Google Gemini (via LangChain's ChatGoogleGenerativeAI), FAISS for vector search, and YouTube's Transcript API.


🚀 Features
✅ Ask questions about any YouTube video with English captions

🧠 RAG-based retrieval using LangChain & FAISS

💡 Quick actions like "Summarize Video" and "Key Points"

📜 Chat history with expand/collapse UI

🎨 Beautifully styled UI using custom Streamlit theming

🔒 Secure environment variable handling with .env

📦 Tech Stack
Feature	Tech Used
Frontend	Streamlit
Backend LLM	Google Gemini via LangChain
Embeddings	GoogleGenerativeAIEmbeddings
RAG Implementation	FAISS + LangChain Runnable Pipelines
Transcript API	youtube-transcript-api
Env Management	python-dotenv

🛠️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/youtube-transcript-qa.git
cd youtube-transcript-qa
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Requirements include:

streamlit

langchain

langchain-google-genai

langchain-community

youtube-transcript-api

python-dotenv

3. Set Environment Variables
Create a .env file in the root directory:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_generative_ai_key
4. Run the App
bash
Copy
Edit
streamlit run app.py
✨ How It Works
🔗 1. Input a YouTube Video
Paste a YouTube link or Video ID into the sidebar.

📜 2. Transcript Extraction
Uses youtube-transcript-api to fetch English subtitles, if available.

🔍 3. Chunking + Embedding + Indexing
The transcript is split into semantic chunks using LangChain's RecursiveCharacterTextSplitter, embedded using Google Embeddings, and indexed using FAISS.

🧠 4. Retrieval-Augmented Generation (RAG)
A custom LangChain RAG pipeline is built using RunnableParallel, PromptTemplate, and StrOutputParser to retrieve relevant transcript sections and generate LLM-powered answers.

💬 5. Ask & Get Answers
Users can interactively ask questions, receive answers, and browse chat history. Quick prompts like summary and key points are also provided.

📸 UI Highlights
🎯 Clear sidebar instructions

📝 Video info block with link and status

💬 Expandable Q&A sections

⚡ Streamlined quick action buttons

📱 Responsive layout and theming

🧪 Sample Questions
"Can you summarize the video?"

"What are the key points discussed?"

"What does the speaker say about [topic]?"

"Who is the target audience for this video?"

🧹 To-Do & Improvements
 Add multilingual transcript support

 Add memory-enabled chat sessions

 Deploy on Streamlit Cloud or Hugging Face Spaces

 Add error tracking/logging system
