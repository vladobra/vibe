import os
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    ListIndex,
    StorageContext,
    Settings,
)

# Load environment variables from .env
load_dotenv()

def main():
    # Read environment variables
    load_dir = os.getenv("LOAD_DIR")
    index_dir = os.getenv("INDEX_FILE")

    if not load_dir:
        raise ValueError("Environment variable LOAD_DIR is not set.")
    if not index_dir:
        raise ValueError("Environment variable INDEX_FILE is not set.")

    print("Loading files in the load directory...")
    print(f"LOAD_DIR = {load_dir}")

    # Read documents from directory
    documents = SimpleDirectoryReader(input_dir=load_dir).load_data()

    # Configure global settings (equivalent to old ServiceContext chunk_size_limit)
    Settings.chunk_size = 8000  # size of text chunks for indexing

    # Build a ListIndex (replacement for GPTListIndex)
    index = ListIndex.from_documents(documents)

    # Persist the index to disk
    # INDEX_FILE is treated here as a directory path to store the index
    storage_context: StorageContext = index.storage_context
    storage_context.persist(persist_dir=index_dir)

    print("Done.")

if __name__ == "__main__":
    main()