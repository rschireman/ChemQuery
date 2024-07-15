import modal
import torch
import os
import fitz
from tqdm.auto import tqdm 
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


# image = (
#     modal.Image.from_registry(
#         "nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.12").apt_install("git", "python3-packaging").run_commands("pip install -i https://pypi.org/simple/ bitsandbytes").pip_install("ase", "tqdm", "huggingface_hub", 
#                                                        "pymupdf", "transformers", "spacy",
#                                                        "sentence_transformers", "llama_index",
#                                                        "flash-attn",
#                                                        "accelerate", "torch")
# )

image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "python3-packaging").pip_install("bitsandbytes", "ase", "tqdm", "huggingface_hub", 
                                                       "pymupdf", "transformers", "spacy",
                                                       "sentence_transformers", "llama_index",
                                                       "accelerate", "torch", "packaging")

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
        item["sentence_chunks"] = split_list.remote(input_list=item["sentences"],
                                            slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    return pages_and_texts




# Split each chunk into its own item
@app.function()
def split_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    
    return pages_and_chunks       

@app.function()
def pages_and_chunks_to_df(pages_and_chunks, min_token_length):
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > int(min_token_length)].to_dict(orient="records")
    return pages_and_chunks_over_min_token_len



@app.function(gpu="any")
def create_embeddings(pages_and_chunks_over_min_token_len):
    embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-mpnet-base-v2", device='cuda:0',
                                      trust_remote_code=True) # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    # Create embeddings on the GPU
    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    print(text_chunks_and_embeddings_df)
    # EMBEDDINGS_DF_SAVE_PATH = "/chemquery/text_chunks_and_embeddings_df.csv"
    # text_chunks_and_embeddings_df.to_csv(EMBEDDINGS_DF_SAVE_PATH, index=False)
    # print(os.listdir("/chemquery"))
    # text_chunks_and_embeddings_df = pd.read_csv("/chemquery/text_chunks_and_embeddings_df.csv")

    # text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to('cuda:0')

    return embeddings


def get_pages_and_chunks(df):
    pages_and_chunks = df.to_dict(orient="records")
    return pages_and_chunks








@app.local_entrypoint()
def main(num_sentence_chunk_size, min_token_length):
    NUM_SENTENCE_CHUNK_SIZE = int(num_sentence_chunk_size)
    pages_and_texts = open_and_read_pdf.remote("/pdfs/crystal23.pdf")
    pages_and_texts = split_pdf.remote(pages_and_texts)
    pages_and_texts = split_text_list.remote(pages_and_texts, NUM_SENTENCE_CHUNK_SIZE)
    pages_and_chunks = split_chunks.remote(pages_and_texts)
    df = pages_and_chunks_to_df.remote(pages_and_chunks, min_token_length)
    embeddings = create_embeddings.remote(df)


