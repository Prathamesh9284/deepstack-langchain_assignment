import os
import json
import re
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
import pickle
import logging
from dotenv import load_dotenv
from utils.prompts import SYSTEM_PROMPT, get_human_prompt
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

EMBEDDINGS_SAVE_PATH = os.path.join(os.getcwd(), "story_embeddings")
MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class Relation(BaseModel):
    name: str = Field(description="Name of the related character")
    relation: str = Field(description="Type of relationship")


class CharacterDetails(BaseModel):
    name: str = Field(description="Name of the character")
    storyTitle: str = Field(description="Title of the story")
    summary: str = Field(description="Brief character summary in 2-3 sentences")
    relations: List[Relation] = Field(description="List of character relationships")
    characterType: str = Field(description="Character type: protagonist/antagonist/side character")

def load_vectorstore():
    """Load the saved vector store and embeddings."""
    vectorstore_path = os.path.join(EMBEDDINGS_SAVE_PATH, "faiss_index")
    config_path = os.path.join(EMBEDDINGS_SAVE_PATH, "embeddings_config.pkl")
    
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    logger.info(f"Loading embeddings model: {config['model_name']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config['model_name'],
        model_kwargs={'device': 'cpu', 'token': HF_TOKEN},
        encode_kwargs=config['encode_kwargs']
    )
    
    logger.info(f"Loading vector store from: {vectorstore_path}")
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore

def extract_character_details(character_name: str) -> dict:
    """Extract character details from stories."""
    logger.info(f"Searching for character: {character_name}")
    
    # Load vector store
    vectorstore = load_vectorstore()
    
    # Search for relevant documents
    query = f"Tell me about the character {character_name}"
    logger.info("Performing similarity search...")
    relevant_docs = vectorstore.similarity_search(query, k=2)
    
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get('source', 'Unknown')
        story_name = Path(source).stem.replace('-', ' ').title() if source != 'Unknown' else 'Unknown Story'
        context_parts.append(f"Story: {story_name}\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    logger.info(f"Found {len(relevant_docs)} relevant documents")
    
    # Initialize Mistral LLM
    logger.info("Initializing Mistral AI...")
    llm = ChatMistralAI(
        model="mistral-medium-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.2
    )
    
    # Create structured output LLM
    structured_llm = llm.with_structured_output(CharacterDetails)
    
    human_prompt = get_human_prompt(character_name, context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": human_prompt}
    ]

    logger.info("Generating character details...")
    character_data = structured_llm.invoke(messages)
    logger.info("Successfully extracted character details")
    return character_data.dict()