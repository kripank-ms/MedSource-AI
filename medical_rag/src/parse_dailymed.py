import json
import re
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

SET_ID = "e087ce3b-8d7f-43a3-8d20-54513da7617d"

OUTPUT_FILE = "medical_rag/data/processed/dailymed_metformin_documents.json"


def remove_namespace(tag):
    return tag.split("}", 1)[-1]


def clean_text(element):
    if element is None:
        return ""

    text = " ".join(element.itertext())
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_section_title(section):
    for child in section:
        if remove_namespace(child.tag) == "title":
            return clean_text(child)

    return ""


def has_child_section(section):
    for child in section:
        if remove_namespace(child.tag) == "section":
            return True

    return False


def parse_dailymed(set_id):

    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"

    response = requests.get(url, timeout=30)

    print("Status:", response.status_code)

    response.raise_for_status()

    root = ET.fromstring(response.content)

    documents = []

    for section in root.iter():

        if remove_namespace(section.tag) != "section":
            continue

        title = get_section_title(section)

        if not title:
            continue

        if has_child_section(section):
            continue

        text = clean_text(section)

        if text.startswith(title):
            text = text[len(title):].strip()

        if not text:
            continue

        document = {
            "text": text,
            "metadata": {
                "source": "DailyMed",
                "set_id": set_id,
                "section": title,
                "url": (
                    "https://dailymed.nlm.nih.gov/dailymed/"
                    f"drugInfo.cfm?setid={set_id}"
                )
            }
        }

        documents.append(document)

    return documents


documents = parse_dailymed(SET_ID)

print("Total sections:", len(documents))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_FILE)