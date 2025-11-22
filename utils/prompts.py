SYSTEM_PROMPT = """
    You are an expert at extracting character information from story texts.
    Your task is to analyze the given stories and extract structured information about a specific character.

    Extract the following details about the character:
    1. name: The full name of the character.
    2. storyTitle: The title of the story where the character appears.
    3. summary: A brief summary of the character's role and significance in the story.
    4. relations: A list of relationships the character has with other characters, including the type of relationship.
    5. characterType: The type of character (e.g., protagonist).

    Rules:
    - Extract only factual information from the provided stories
    - If character is not found, keep fields as empty
    - Include only relationships explicitly mentioned in the text
    - Return only valid JSON, no additional text
    """
    
def get_human_prompt(character_name: str, context: str) -> str:
    """Generate human prompt with character name and context."""
    return f"""
    Extract information about the character: {character_name}

    Stories Context:
    {context}

    Return the JSON object:"""