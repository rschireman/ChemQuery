import modal
import torch
import os
import fitz
from tqdm.auto import tqdm 
from spacy.lang.en import English # see https://spacy.io/usage for install instructions


image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel").apt_install("git").pip_install("ase", "tqdm", "huggingface_hub", 
                                                       "pymupdf", "transformers", "spacy",
                                                       "sentence_transformers", "llama_index",
                                                       "flash-attn",
                                                       "accelerate")
)

volume = modal.Volume.from_name("chemquery", create_if_missing=True)

app = modal.App(name="ChemQuery",image=image)

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
@app.function(mounts=[modal.Mount.from_local_dir("./pdf_files/", remote_path="./pdfs")], volumes={"/chemquery": volume})
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        cleaned_text = text.replace("\n", " ").strip()
        pages_and_texts.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(cleaned_text),
                                "page_word_count": len(cleaned_text.split(" ")),
                                "page_sentence_count_raw": len(cleaned_text.split(". ")),
                                "page_token_count": len(cleaned_text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": cleaned_text})
    volume.commit()    
    return pages_and_texts


@app.function()
def split_pdf(pages_and_texts):
    nlp = English()
    # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        print(item)
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])
    
    return pages_and_texts    


# Define split size to turn groups of sentences into chunks


# Create a function that recursively splits a list into desired sizes
@app.function()
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

@app.function()
def split_text_list(pages_and_texts, num_sentence_chunk_size):
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        print(item)
        item["sentence_chunks"] = split_list.remote(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    return pages_and_texts






@app.local_entrypoint()
def main(num_sentence_chunk_size):

    NUM_SENTENCE_CHUNK_SIZE = int(num_sentence_chunk_size)
    pages_and_texts = open_and_read_pdf.remote("/pdfs/crystal23.pdf")
    pages_and_texts = split_pdf.remote(pages_and_texts)
    pages_and_texts = split_text_list.remote(pages_and_texts, NUM_SENTENCE_CHUNK_SIZE)

