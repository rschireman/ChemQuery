import modal
import torch
import os
import fitz
from tqdm.auto import tqdm 


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

@app.function(gpu="any")
def test_container():
    os.system("nvidia-smi")

@app.function(gpu="any")
def get_gpu_mem():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    return gpu_memory_gb


# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
@app.function(gpu="any", mounts=[modal.Mount.from_local_dir("./pdf_files/", remote_path="./pdfs")], volumes={"/chemquery": volume})
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

    
    print(os.listdir("/pdfs"))
    
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
                                "text": text})
    return pages_and_texts





@app.local_entrypoint()
def main():
    # test_container.remote()
    # print(f"Available GPU memory: {get_gpu_mem.remote()} GB")    
    open_and_read_pdf.remote("/pdfs/crystal23.pdf")