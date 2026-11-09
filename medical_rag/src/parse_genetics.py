import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from collections import Counter


INPUT_FILE = "medical_rag/data/raw/ghr-summaries.xml"
OUTPUT_FILE = "medical_rag/data/processed/medlineplus_genetics_documents.json"


def strip_namespace(root):
    """
    Remove XML namespaces so we can use simple tag names
    such as 'name', 'text-list', etc.
    """

    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]

    return root


def clean_html(element):
    """
    Convert HTML inside an XML element into normal text.
    """

    if element is None:
        return ""

    html_text = ET.tostring(
        element,
        encoding="unicode"
    )

    soup = BeautifulSoup(html_text, "html.parser")

    return soup.get_text(
        separator="\n",
        strip=True
    )


def get_text(element, tag):
    """
    Get simple text from a child element.
    """

    child = element.find(tag)

    if child is not None and child.text:
        return child.text.strip()

    return ""


def get_synonyms(record):
    """
    Extract synonyms.
    """

    synonyms = []

    synonym_list = record.find("synonym-list")

    if synonym_list is not None:

        for synonym in synonym_list.findall("synonym"):

            if synonym.text:
                synonyms.append(
                    synonym.text.strip()
                )

    return synonyms


def get_db_keys(record):
    """
    Extract database references.
    """

    db_keys = []

    db_key_list = record.find("db-key-list")

    if db_key_list is not None:

        for db_key in db_key_list.findall("db-key"):

            db = get_text(db_key, "db")
            key = get_text(db_key, "key")

            if db or key:

                db_keys.append({
                    "database": db,
                    "key": key
                })

    return db_keys


def get_text_sections(record):
    """
    Extract descriptive text sections.
    """

    sections = []

    text_list = record.find("text-list")

    if text_list is None:
        return sections

    for text_element in text_list.findall("text"):

        role = get_text(
            text_element,
            "text-role"
        )

        html_element = text_element.find("html")

        text = clean_html(html_element)

        if text:

            sections.append({
                "role": role,
                "text": text
            })

    return sections


def get_related_genes(record):
    """
    Extract genes related to a genetic condition.
    """

    genes = []

    gene_list = record.find(
        "related-gene-list"
    )

    if gene_list is None:
        return genes

    for gene in gene_list.findall(
        "related-gene"
    ):

        genes.append({
            "symbol": get_text(
                gene,
                "gene-symbol"
            ),
            "url": get_text(
                gene,
                "ghr-page"
            )
        })

    return genes


def get_related_chromosomes(record):
    """
    Extract chromosomes related to a condition.
    """

    chromosomes = []

    gene_list = record.find(
        "related-gene-list"
    )

    if gene_list is None:
        return chromosomes

    for chromosome in gene_list.findall(
        "related-chromosome"
    ):

        chromosomes.append({
            "name": get_text(
                chromosome,
                "name"
            ),
            "url": get_text(
                chromosome,
                "ghr-page"
            )
        })

    return chromosomes


def get_related_conditions(record):
    """
    Extract conditions related to genes/chromosomes.
    """

    conditions = []

    condition_list = record.find(
        "related-health-condition-list"
    )

    if condition_list is None:
        return conditions

    for condition in condition_list:

        name = get_text(
            condition,
            "name"
        )

        url = get_text(
            condition,
            "ghr-page"
        )

        if name or url:

            conditions.append({
                "name": name,
                "url": url
            })

    return conditions


def get_inheritance_patterns(record):
    """
    Extract inheritance patterns.
    """

    patterns = []

    pattern_list = record.find(
        "inheritance-pattern-list"
    )

    if pattern_list is None:
        return patterns

    for pattern in pattern_list.findall(
        "inheritance-pattern"
    ):

        code = get_text(
            pattern,
            "code"
        )

        memo = get_text(
            pattern,
            "memo"
        )

        patterns.append({
            "code": code,
            "memo": memo
        })

    return patterns


def parse_record(record):

    tag = record.tag

    # --------------------------------
    # Genetic condition
    # --------------------------------

    if tag == "health-condition-summary":

        record_type = "health_condition"

        name = get_text(
            record,
            "name"
        )

        ghr_page = get_text(
            record,
            "ghr-page"
        )

        text_sections = get_text_sections(
            record
        )

        synonyms = get_synonyms(
            record
        )

        inheritance_patterns = (
            get_inheritance_patterns(
                record
            )
        )

        related_genes = get_related_genes(
            record
        )

        related_chromosomes = (
            get_related_chromosomes(
                record
            )
        )

        db_keys = get_db_keys(
            record
        )

        searchable_parts = []

        if name:
            searchable_parts.append(name)

        if synonyms:
            searchable_parts.append(
                "Also called: "
                + ", ".join(synonyms)
            )

        for section in text_sections:

            if section["role"]:
                searchable_parts.append(
                    section["role"]
                    + ": "
                    + section["text"]
                )
            else:
                searchable_parts.append(
                    section["text"]
                )

        searchable_text = "\n\n".join(
            searchable_parts
        )

        metadata = {
            "record_type": record_type,
            "id": record.attrib.get("id"),
            "title": name,
            "url": ghr_page,
            "synonyms": synonyms,
            "inheritance_patterns":
                inheritance_patterns,
            "related_genes":
                related_genes,
            "related_chromosomes":
                related_chromosomes,
            "db_keys": db_keys,
            "reviewed": get_text(
                record,
                "reviewed"
            ),
            "published": get_text(
                record,
                "published"
            )
        }

    # --------------------------------
    # Gene
    # --------------------------------

    elif tag == "gene-summary":

        record_type = "gene"

        symbol = get_text(
            record,
            "gene-symbol"
        )

        name = get_text(
            record,
            "name"
        )

        ghr_page = get_text(
            record,
            "ghr-page"
        )

        text_sections = get_text_sections(
            record
        )

        synonyms = get_synonyms(
            record
        )

        related_conditions = (
            get_related_conditions(
                record
            )
        )

        searchable_parts = []

        if symbol:
            searchable_parts.append(
                "Gene symbol: " + symbol
            )

        if name:
            searchable_parts.append(
                "Gene name: " + name
            )

        if synonyms:
            searchable_parts.append(
                "Also called: "
                + ", ".join(synonyms)
            )

        for section in text_sections:

            if section["role"]:
                searchable_parts.append(
                    section["role"]
                    + ": "
                    + section["text"]
                )
            else:
                searchable_parts.append(
                    section["text"]
                )

        searchable_text = "\n\n".join(
            searchable_parts
        )

        metadata = {
            "record_type": record_type,
            "id": record.attrib.get("id"),
            "title": name or symbol,
            "gene_symbol": symbol,
            "url": ghr_page,
            "synonyms": synonyms,
            "related_conditions":
                related_conditions,
            "reviewed": get_text(
                record,
                "reviewed"
            ),
            "published": get_text(
                record,
                "published"
            )
        }

    # --------------------------------
    # Chromosome
    # --------------------------------

    elif tag == "chromosome-summary":

        record_type = "chromosome"

        name = get_text(
            record,
            "name"
        )

        ghr_page = get_text(
            record,
            "ghr-page"
        )

        text_sections = get_text_sections(
            record
        )

        related_conditions = (
            get_related_conditions(
                record
            )
        )

        searchable_parts = []

        if name:
            searchable_parts.append(name)

        for section in text_sections:

            if section["role"]:
                searchable_parts.append(
                    section["role"]
                    + ": "
                    + section["text"]
                )
            else:
                searchable_parts.append(
                    section["text"]
                )

        searchable_text = "\n\n".join(
            searchable_parts
        )

        metadata = {
            "record_type": record_type,
            "id": record.attrib.get("id"),
            "title": name,
            "url": ghr_page,
            "related_conditions":
                related_conditions,
            "reviewed": get_text(
                record,
                "reviewed"
            ),
            "published": get_text(
                record,
                "published"
            )
        }

    # --------------------------------
    # mtDNA
    # --------------------------------

    elif tag == "mtdna-summary":

        record_type = "mtdna"

        name = get_text(
            record,
            "name"
        )

        ghr_page = get_text(
            record,
            "ghr-page"
        )

        text_sections = get_text_sections(
            record
        )

        related_conditions = (
            get_related_conditions(
                record
            )
        )

        searchable_parts = []

        if name:
            searchable_parts.append(name)

        for section in text_sections:

            if section["role"]:
                searchable_parts.append(
                    section["role"]
                    + ": "
                    + section["text"]
                )
            else:
                searchable_parts.append(
                    section["text"]
                )

        searchable_text = "\n\n".join(
            searchable_parts
        )

        metadata = {
            "record_type": record_type,
            "id": record.attrib.get("id"),
            "title": name,
            "url": ghr_page,
            "related_conditions":
                related_conditions,
            "reviewed": get_text(
                record,
                "reviewed"
            ),
            "published": get_text(
                record,
                "published"
            )
        }

    else:
        return None

    if not searchable_text.strip():
        return None

    return {
        "text": searchable_text,
        "metadata": metadata
    }


def parse_genetics(file_path):

    tree = ET.parse(file_path)

    root = tree.getroot()

    # Important:
    # Remove XML namespace before searching tags.
    root = strip_namespace(root)

    documents = []

    for record in root:

        parsed = parse_record(record)

        if parsed is not None:
            documents.append(parsed)

    return documents


# --------------------------------
# Run parser
# --------------------------------

documents = parse_genetics(
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


# --------------------------------
# Validation
# --------------------------------

record_types = Counter(
    document["metadata"]["record_type"]
    for document in documents
)

empty_documents = sum(
    1
    for document in documents
    if not document["text"].strip()
)


print()
print("================================")
print("GENETICS PARSING COMPLETE")
print("================================")

print(
    "Total documents:",
    len(documents)
)

print(
    "Record types:",
    dict(record_types)
)

print(
    "Empty documents:",
    empty_documents
)

print(
    "Output:",
    OUTPUT_FILE
)

print()
print("First document:")
print(
    json.dumps(
        documents[0],
        indent=2,
        ensure_ascii=False
    )
)