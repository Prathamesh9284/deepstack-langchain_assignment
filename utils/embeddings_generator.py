import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import pickle
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'    
)

logger = logging.getLogger(__name__)

STORIES_FOLDER = os.path.join(os.getcwd(), "stories")
EMBEDDINGS_SAVE_PATH = os.path.join(os.getcwd(), "story_embeddings")
EMBEDDING_MODEL = "google/embeddinggemma-300m"  
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def load_stories(folder_path):
    """Load all text files from the stories folder."""
    documents = []
    story_folder = Path(folder_path)
    
    if not story_folder.exists():
        raise FileNotFoundError(f"Stories folder '{folder_path}' not found!")
    
    txt_files = list(story_folder.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{folder_path}'")
    
    logger.info(f"Found {len(txt_files)} story files")
    
    for file_path in txt_files:
        logger.info(f"Loading: {file_path.name}")
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents.extend(loader.load())
    
    return documents

def create_embeddings(documents):
    """Create embeddings for the documents."""
    logger.info(f"\nInitializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu', 'token': HF_TOKEN},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Using documents as single chunks (no splitting)...")
    logger.info(f"Total documents: {len(documents)}")
    
    # Creating vector store directly from documents
    logger.info("Generating embeddings and creating vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore, embeddings

def save_embeddings(vectorstore, embeddings):
    """Save the vector store and embeddings."""
    os.makedirs(EMBEDDINGS_SAVE_PATH, exist_ok=True)
    
    # Saving FAISS vector store
    vectorstore_path = os.path.join(EMBEDDINGS_SAVE_PATH, "faiss_index")
    logger.info(f"Saving vector store to: {vectorstore_path}")
    vectorstore.save_local(vectorstore_path)
    
    # Saving embeddings configuration
    config_path = os.path.join(EMBEDDINGS_SAVE_PATH, "embeddings_config.pkl")
    logger.info(f"Saving embeddings config to: {config_path}")
    with open(config_path, 'wb') as f:
        pickle.dump({
            'model_name': EMBEDDING_MODEL,
            'model_kwargs': embeddings.model_kwargs,
            'encode_kwargs': embeddings.encode_kwargs
        }, f)
    
    logger.info("Embeddings saved successfully!!!")

def main():
    logger.info("=" * 60)
    logger.info("STORY EMBEDDINGS GENERATOR")
    logger.info("=" * 60)
    
    # Loading stories
    documents = load_stories(STORIES_FOLDER)
    
    # Creating embeddings
    vectorstore, embeddings = create_embeddings(documents)
    
    # Saving embeddings
    save_embeddings(vectorstore, embeddings)
    
    logger.info("=" * 60)
    logger.info("PROCESS COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()