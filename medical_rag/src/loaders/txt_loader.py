def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    document = {
        "text":text,
        "metadata":{
            "source_type":"txt",
            "file_name": file_path
        }
    }
    return document

document = load_txt("medical_rag/data/raw/sample.txt")
