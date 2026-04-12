import json
import csv
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample Bible verses in English and Spanish (ESV and RVR60)
SAMPLE_VERSES = [
    {
        "book": "Genesis",
        "chapter": 1,
        "verse": 1,
        "en": "In the beginning, God created the heavens and the earth.",
        "es": "En el principio creó Dios los cielos y la tierra.",
        "sw": "Mwanzo Mungu akaumba mbingu na ardhi.",
    },
    {
        "book": "Genesis",
        "chapter": 1,
        "verse": 2,
        "en": "The earth was without form and void, and darkness was over the face of the deep.",
        "es": "Y la tierra estaba desordenada y vacía, y las tinieblas cubrían la faz del abismo.",
        "sw": "Nchi ilikuwa bila sura na tupu, na giza likakuwa juu ya uso wa kirefu.",
    },
    {
        "book": "Psalm",
        "chapter": 23,
        "verse": 1,
        "en": "The Lord is my shepherd; I shall not want.",
        "es": "Jehová es mi pastor; nada me faltará.",
        "sw": "Mungu ni mchumi wangu, sitakosa chochote.",
    },
    {
        "book": "Psalm",
        "chapter": 23,
        "verse": 2,
        "en": "He makes me lie down in green pastures.",
        "es": "En lugares de delicados pastos me hará descansar.",
        "sw": "Anifanya nilete katika malisho yenye majani mema.",
    },
    {
        "book": "John",
        "chapter": 3,
        "verse": 16,
        "en": "For God so loved the world, that he gave his only Son.",
        "es": "Porque de tal manera amó Dios al mundo, que ha dado a su Hijo unigénito.",
        "sw": "Kwa kuwa Mungu akamkamatia kidunia hivyo, hata akamkasikia Mwanae wa pekee.",
    },
    {
        "book": "Matthew",
        "chapter": 6,
        "verse": 9,
        "en": "Our Father in heaven, hallowed be your name.",
        "es": "Padre nuestro que estás en los cielos, santificado sea tu nombre.",
        "sw": "Baba yetu ulijaye juu ya angani, jina lako na litukuzwe.",
    },
    {
        "book": "Luke",
        "chapter": 1,
        "verse": 26,
        "en": "In the sixth month the angel Gabriel was sent from God.",
        "es": "Al sexto mes el ángel Gabriel fue enviado por Dios.",
        "sw": "Katika mwezi wa sita, malaika Gabrieli alitumwa na Mungu.",
    },
    {
        "book": "Romans",
        "chapter": 3,
        "verse": 23,
        "en": "All have sinned and fall short of the glory of God.",
        "es": "Por cuanto todos pecaron, y están destituidos de la gloria de Dios.",
        "sw": "Kwa kuwa wote wametenda dhambi, na wamesinyanganywa mahaba ya Mungu.",
    },
    {
        "book": "1 John",
        "chapter": 4,
        "verse": 7,
        "en": "Beloved, let us love one another, for love is from God.",
        "es": "Amados, amémonos unos a otros; porque el amor es de Dios.",
        "sw": "Wapenzi, tumwaeni mmoja mmoja; kwa kuwa upendo ni wa Mungu.",
    },
    {
        "book": "Revelation",
        "chapter": 21,
        "verse": 1,
        "en": "Then I saw a new heaven and a new earth.",
        "es": "Vi un cielo nuevo y una tierra nueva.",
        "sw": "Kisha niliona angani mpya na ardhi mpya.",
    },
]


def generate_sample_data(output_dir: Path = None):
    """Generate sample Bible verse data in multiple formats"""
    
    output_dir = output_dir or Path("./data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate JSON files for each language
    logger.info("Generating JSON files...")
    
    for lang, lang_key in [("English", "en"), ("Spanish", "es"), ("Swahili", "sw")]:
        json_path = output_dir / f"{lang_key}_verses.json"
        
        verses = []
        for verse_data in SAMPLE_VERSES:
            if lang_key in verse_data:
                verses.append({
                    "book": verse_data["book"],
                    "chapter": verse_data["chapter"],
                    "verse": verse_data["verse"],
                    "text": verse_data[lang_key],
                    "language": lang_key,
                })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(verses, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(verses)} verses to {json_path}")
    
    # 2. Generate parallel corpus (JSONL format)
    logger.info("Generating parallel corpus...")
    
    # English-Spanish
    jsonl_path = output_dir / "en_es_verses.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for verse in SAMPLE_VERSES:
            item = {
                "source": verse["en"],
                "target": verse["es"],
                "source_lang": "eng_Latn",
                "target_lang": "spa_Latn",
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved parallel corpus to {jsonl_path}")
    
    # English-Swahili
    jsonl_path = output_dir / "en_sw_verses.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for verse in SAMPLE_VERSES:
            item = {
                "source": verse["en"],
                "target": verse["sw"],
                "source_lang": "eng_Latn",
                "target_lang": "swh_Latn",
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved parallel corpus to {jsonl_path}")
    
    # 3. Generate CSV format
    logger.info("Generating CSV files...")
    
    csv_path = output_dir / "verses_parallel.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["book", "chapter", "verse", "english", "spanish", "swahili"])
        writer.writeheader()
        
        for verse in SAMPLE_VERSES:
            writer.writerow({
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "english": verse["en"],
                "spanish": verse["es"],
                "swahili": verse["sw"],
            })
    
    logger.info(f"Saved CSV to {csv_path}")
    
    return output_dir


def create_test_dataset(output_path: Path = None, num_samples: int = 100):
    """Create a larger test dataset by repeating sample verses"""
    
    output_path = output_path or Path("./data/verse_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating test dataset with {num_samples} samples...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            verse = SAMPLE_VERSES[i % len(SAMPLE_VERSES)]
            item = {
                "source": verse["en"],
                "target": verse["es"],
                "source_lang": "eng_Latn",
                "target_lang": "spa_Latn",
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "sample_id": i,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved test dataset to {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate sample data
    data_dir = generate_sample_data()
    print(f"\nSample data generated in {data_dir}")
    
    # Create test dataset
    test_path = create_test_dataset(num_samples=50)
    print(f"Test dataset created at {test_path}")
