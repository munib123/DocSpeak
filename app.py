import gradio as gr
import os
from rag import RAGSystem
from dotenv import load_dotenv
from gtts import gTTS
from docx2pdf import convert

# Load environment variables from .env file
load_dotenv()

# Initialize the RAG system with default settings
pdf_dir = "material"
db_dir = "chroma_db"
gemini_api_key = os.getenv("GEMINI_API_KEY")
rag_system = RAGSystem(pdf_dir=pdf_dir, gemini_api_key=gemini_api_key, db_directory=db_dir)

# Function to handle file upload and process the uploaded file
def upload_and_process(file):
    if file is not None:
        uploaded_file_path = file.name
        ext = os.path.splitext(uploaded_file_path)[1].lower()
        if ext == ".docx":
            # Convert DOCX to PDF
            pdf_path = uploaded_file_path.replace(".docx", ".pdf")
            convert(uploaded_file_path, pdf_path)
            rag_system.pdf_dir = os.path.dirname(pdf_path)
        else:
            rag_system.pdf_dir = os.path.dirname(uploaded_file_path)
        rag_system.process_documents()
        return "File uploaded and processed successfully."
    return "No file uploaded."

# Updated function to handle user queries
def ask_query(query):
    if query.strip():
        response = rag_system.generate_response(query)
        # Append the query and response to the conversation history
        rag_system.conversation_history.append({"user": query, "system": response})
        audio_path = text_to_speech(response)
        return response, audio_path
    return "Please enter a valid query.", None

# Function to convert text to speech and return audio file path
def text_to_speech(response):
    tts = gTTS(response)
    audio_path = "response_audio.mp3"
    tts.save(audio_path)
    return audio_path

# Function to clear the chat history
def clear_chat():
    rag_system.conversation_history = []
    return "Chat history cleared."

# Updated function to download the chat history
def download_chat():
    chat_history = "\n".join([f"User: {entry['user']}\nSystem: {entry['system']}" for entry in rag_system.conversation_history])
    file_path = "chat_history.txt"
    with open(file_path, "w") as file:
        file.write(chat_history)
    return file_path

# --- Custom CSS for modern look ---
custom_css = '''
body { font-family: 'Roboto', 'Open Sans', Arial, sans-serif; }
.gradio-container { background: linear-gradient(135deg, #f8fafc 0%, #e0f7fa 100%); }
#rag-title { font-size: 2.2rem; font-weight: 700; color: #1e293b; letter-spacing: 1px; display: flex; align-items: center; gap: 0.5em; }
#rag-title img { height: 2.2rem; vertical-align: middle; }
.gr-box { border-radius: 12px !important; box-shadow: 0 2px 12px 0 rgba(16, 42, 67, 0.06); border: 1px solid #e0e7ef; }
.gr-button { background: #14b8a6; color: #fff; border-radius: 8px; font-weight: 600; font-size: 1rem; padding: 0.7em 1.5em; transition: background 0.2s, transform 0.2s; }
.gr-button:hover, .gr-button:focus { background: #0d9488; transform: scale(1.04); }
.gr-text-input, .gr-textbox { border-radius: 8px; border: 1.5px solid #cbd5e1; background: #fff; font-size: 1.05rem; }
.gr-text-input:focus, .gr-textbox:focus { border-color: #14b8a6; box-shadow: 0 0 0 2px #99f6e4; }
.gr-audio { border-radius: 8px; background: #f1f5f9; }
.gr-file { border-radius: 8px; border: 1.5px dashed #14b8a6; background: #f0fdfa; }
.gr-markdown { color: #334155; }
.fade-in { animation: fadeIn 0.7s; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@media (max-width: 700px) {
  #rag-title { font-size: 1.3rem; }
  .gradio-container { padding: 0.5em; }
}
'''

# --- Gradio UI with modern design ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as ui:
    # Branding row with logo and title
    with gr.Row():
        gr.Markdown("""
        <div id='rag-title'>
            <img src='https://img.icons8.com/color/48/000000/artificial-intelligence.png' alt='RAG Logo' />
            RAG System UI
        </div>
        """, elem_id="rag-title")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            file_input = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"], elem_classes=["gr-file"])
            upload_button = gr.Button("Upload & Process File", elem_classes=["gr-button"])
        with gr.Column(scale=2, min_width=400):
            query_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...", lines=1, elem_classes=["gr-text-input"])
            response_output = gr.Textbox(label="RAG Response", lines=6, interactive=False, elem_classes=["gr-textbox", "fade-in"])
            play_button = gr.Audio(label="Play Response", interactive=False, elem_classes=["gr-audio"])

    with gr.Row():
        clear_button = gr.Button("Clear Chat History", elem_classes=["gr-button"])
        download_button = gr.Button("Download Chat History", elem_classes=["gr-button"])
        download_file = gr.File(label="Download Chat File", elem_classes=["gr-file"])

    # Bind functions to UI components
    upload_button.click(upload_and_process, inputs=file_input, outputs=None)
    query_input.submit(ask_query, inputs=query_input, outputs=[response_output, play_button])
    clear_button.click(clear_chat, inputs=None, outputs=None)
    download_button.click(download_chat, inputs=None, outputs=download_file)

# Launch the Gradio app
if __name__ == "__main__":
    ui.launch()
