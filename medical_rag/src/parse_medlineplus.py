import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


INPUT_FILE = "medical_rag/data/raw/mplus_topics_2026-08-20.xml"
OUTPUT_FILE = "medical_rag/data/processed/medlineplus_documents.json"


def clean_html(html_text):
    """Convert HTML inside full-summary into clean text."""

    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    return soup.get_text(
        separator="\n",
        strip=True
    )


def parse_medlineplus(file_path):

    tree = ET.parse(file_path)
    root = tree.getroot()

    documents = []

    health_topics = root.findall(".//health-topic")

    print("Health topics found:", len(health_topics))

    for topic in health_topics:



        topic_id = topic.attrib.get("id")
        title = topic.attrib.get("title")
        url = topic.attrib.get("url")
        language = topic.attrib.get("language")
        date_created = topic.attrib.get("date-created")
        meta_desc = topic.attrib.get("meta-desc")


        also_called = []

        for element in topic.findall("also-called"):

            if element.text:
                also_called.append(
                    element.text.strip()
                )



        summary_element = topic.find("full-summary")

        if summary_element is not None:

            summary_html = summary_element.text or ""

            summary_text = clean_html(
                summary_html
            )

        else:

            summary_text = ""



        groups = []

        for group in topic.findall("group"):

            groups.append({
                "id": group.attrib.get("id"),
                "name": group.text.strip()
                if group.text else None,
                "url": group.attrib.get("url")
            })



        mesh_terms = []

        mesh_heading = topic.find("mesh-heading")

        if mesh_heading is not None:

            for descriptor in mesh_heading.findall(
                "descriptor"
            ):

                mesh_terms.append({
                    "id": descriptor.attrib.get("id"),
                    "name": descriptor.text.strip()
                    if descriptor.text else None
                })


        see_references = []

        for reference in topic.findall(
            "see-reference"
        ):

            if reference.text:

                see_references.append(
                    reference.text.strip()
                )


        # -----------------------------
        # Primary institute
        # -----------------------------

        institute = topic.find(
            "primary-institute"
        )

        primary_institute = None

        if institute is not None:

            primary_institute = {
                "name": institute.text.strip()
                if institute.text else None,

                "url": institute.attrib.get("url")
            }


        language_topics = []

        for element in topic.findall(
            "language-mapped-topic"
        ):

            language_topics.append({
                "language": element.attrib.get(
                    "language"
                ),

                "title": element.text.strip()
                if element.text else None,

                "url": element.attrib.get("url"),

                "id": element.attrib.get("id")
            })


        related_sites = []

        for site in topic.findall("site"):

            related_sites.append({
                "title": site.attrib.get("title"),

                "url": site.attrib.get("url"),

                "language_mapped_url":
                    site.attrib.get(
                        "language-mapped-url"
                    ),

                "information_category":
                    site.findtext(
                        "information-category"
                    ),

                "organization":
                    site.findtext(
                        "organization"
                    ),

                "standard_description":
                    site.findtext(
                        "standard-description"
                    )
            })


        other_languages = []

        for element in topic.findall(
            "other-language"
        ):

            other_languages.append({
                "language":
                    element.attrib.get(
                        "vernacular-name"
                    ),

                "url":
                    element.attrib.get("url")
            })


        text_parts = []

        if title:
            text_parts.append(title)

        if also_called:

            text_parts.append(
                "Also called: "
                + ", ".join(also_called)
            )

        if see_references:

            text_parts.append(
                "See also: "
                + ", ".join(see_references)
            )

        if summary_text:

            text_parts.append(
                summary_text
            )

        searchable_text = "\n\n".join(
            text_parts
        )


        document = {

            "text": searchable_text,

            "metadata": {

                "source": "MedlinePlus",

                "id": topic_id,

                "title": title,

                "meta_description": meta_desc,

                "url": url,

                "language": language,

                "date_created": date_created,

                "also_called": also_called,

                "groups": groups,

                "mesh_terms": mesh_terms,

                "see_references": see_references,

                "primary_institute":
                    primary_institute,

                "language_topics":
                    language_topics,

                "other_languages":
                    other_languages,

                "related_sites":
                    related_sites
            }
        }

        documents.append(document)


    return documents


documents = parse_medlineplus(
    INPUT_FILE
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        documents,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print("Parsing complete")
print()
print("Documents:", len(documents))
print("Output:", OUTPUT_FILE)