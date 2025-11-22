# Character Details Extractor

A Python application that extracts structured character information from story texts using embeddings and AI. The system uses FAISS for vector storage, HuggingFace embeddings for semantic search, and Mistral AI for intelligent character detail extraction.

## Features

- **Embedding Generation**: Converts story texts into vector embeddings for efficient semantic search
- **Character Extraction**: Extracts detailed character information including:
  - Character name
  - Story title
  - Character summary
  - Relationships with other characters
  - Character type (protagonist/antagonist/side character)

## Project Structure

```
├── stories/                        # Directory containing story text files
│   ├── the-schoolmistress.txt
│   └── ...
├── utils/
│   ├── embeddings_generator.py     # Generates and saves embeddings
│   ├── extractor.py                # Character extraction logic
│   └── prompts.py                  # System and user prompts
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
└── README.md                   
```

## Prerequisites

- Python 3.12+
- Mistral AI API key
- Hugging Face token (for embedding model access)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Deepstack Software"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   Windows (PowerShell):
   ```bash
   .\venv\Scripts\Activate
   ```
   
   Windows (Command Prompt):
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   
   Create a `.env` file in the root directory with:
   ```env
   MISTRAL_KEY=your_mistral_api_key_here
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

## Usage

### Step 1: Generate Embeddings

First, generate embeddings from your story files:

```bash
python utils/embeddings_generator.py
```

This will:
- Load all `.txt` files from the `stories/` folder
- Generate embeddings using the `google/embeddinggemma-300m` model
- Save the FAISS vector store to `story_embeddings/`

### Step 2: Extract Character Details

Run the main script to extract character information:

```bash
python main.py
```

When prompted, enter a character name (e.g., "Marya Vassilyevna").

The script will:
- Search for the character in the story embeddings
- Extract detailed information using Mistral AI
- Display the results in JSON format
- Save the output to `result.json`

## Example Output

```json
{
  "name": "Marya Vassilyevna",
  "storyTitle": "The Schoolmistress",
  "summary": "Marya Vassilyevna is a schoolmistress who has been teaching for thirteen years in a remote village. She lives a lonely, difficult life, feeling disconnected from her past and trapped in her routine.",
  "relations": [
    {
      "name": "Semyon",
      "relation": "Cart driver"
    },
    {
      "name": "Hanov",
      "relation": "Landowner and former school examiner"
    }
  ],
  "characterType": "protagonist"
}
```

## Dependencies

- `langchain` - LangChain framework
- `langchain-community` - Community integrations
- `langchain-huggingface` - HuggingFace integration
- `langchain-mistralai` - Mistral AI integration
- `sentence-transformers` - Sentence embedding models
- `transformers` - Hugging Face transformers
- `faiss-cpu` - Vector similarity search
- `torch` - PyTorch for embeddings
- `numpy` - Numerical operations
- `huggingface-hub` - HuggingFace model hub access

## Edge Cases

- **Character Not Found**: If the character is not found in any story, the system returns empty fields with a warning message
- **Multiple Occurrences**: The system searches across all stories and returns details from the most relevant matches
- **Ambiguous Names**: Returns information from the most relevant story context
