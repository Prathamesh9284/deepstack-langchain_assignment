import sys
import json
import logging
from utils.extractor import extract_character_details

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():    
    character_name = input("Enter the character name to extract details for: ").strip()
    
    logger.info("=" * 60)
    logger.info("CHARACTER DETAILS EXTRACTOR")
    logger.info("=" * 60)
    
    # Extract character details
    character_data = extract_character_details(character_name)
    
    if character_data['name'] == '':
        logger.warning(f"Character '{character_name}' not found in the stories.")
        return
    
    # Convert to JSON string
    json_output = json.dumps(character_data, indent=2)
    
    # Display JSON
    print("\n" + "=" * 60)
    print("CHARACTER DETAILS:")
    print("=" * 60)
    print(json_output)
    print("=" * 60)
    
    # Save to file
    filename = f"result.json"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_output)
    
    logger.info(f"Output saved to: {filename}")
    logger.info("Extraction completed successfully!")

if __name__ == "__main__":
    main()