import os

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

# ---------------------------------------------------------
# Global objects
# ---------------------------------------------------------
index = None
chat_engine = None


# ---------------------------------------------------------
# Index + chat engine initialization
# ---------------------------------------------------------
def initialise_index():
    global index, chat_engine

    load_dotenv()

    # Where the data and index are stored
    data_dir = os.getenv("LOAD_DIR", "data")
    index_dir = os.getenv("INDEX_FILE", "storage")

    # Configure LlamaIndex global settings (replaces ServiceContext)
    Settings.llm = OpenAI(model="gpt-5")  # requires OPENAI_API_KEY
    # Optional: adjust if you want different chunking
    Settings.chunk_size = 8000

    # Optional: you can also explicitly set an embedding model, e.g.:
    # from llama_index.embeddings.openai import OpenAIEmbedding
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Reranker
    rerank = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large",
        top_n=4,
    )

    # Load or build index
    if os.path.exists(index_dir) and os.listdir(index_dir):
        # Load existing index from disk
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        # Build new index
        os.makedirs(index_dir, exist_ok=True)
        data = SimpleDirectoryReader(input_dir=data_dir).load_data()
        index = VectorStoreIndex.from_documents(data)
        index.storage_context.persist(persist_dir=index_dir)

    # Chat memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=32000)

    # Create chat engine (no ServiceContext arg in latest versions)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=10,
        node_postprocessors=[rerank],
        memory=memory,
        context_prompt=(
            "You are a chatbot, called Jack. "
            "Here are the relevant documents for the context:\n"
            "{context_str}\n"
            "Instruction: Based on the above documents, provide an answer for "
            "the user question below. Make the answer precise and very short. "
            "Answer as short as possible."
        ),
    )


# ---------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------
# get path for GUI
gui_dir = os.path.join(os.path.dirname(__file__), "gui")
if not os.path.exists(gui_dir):
    gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui")

server = Flask(__name__, static_folder=gui_dir, template_folder=gui_dir)

# initialise index on startup
initialise_index()


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@server.route("/")
def landing():
    return render_template("index.html")


@server.route("/query", methods=["POST"])
def query():
    global chat_engine

    data = request.get_json(force=True)
    user_query = data.get("user_query", "")
    response = chat_engine.chat(user_query)
    return jsonify(str(response))


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set in your environment or .env file.
    server.run(host="0.0.0.0", port=8000)