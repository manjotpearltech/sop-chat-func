
import logging, os, io, json
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import pandas as pd
from utils import BS, ACCOUNT, get_text_from_pdf, get_text_from_docx, get_text_from_pptx, smart_chunk, embed_texts, save_parquet

def main(inputblob: func.InputStream):
    name = inputblob.name  # container/path
    blob_name = name.split("/",1)[1]
    logging.info(f"[process_upload] Triggered for {blob_name}, size={inputblob.length}")

    # Download bytes
    content = inputblob.read()
    ext = os.path.splitext(blob_name)[1].lower()

    # Extract text
    if ext == ".pdf":
        text = get_text_from_pdf(content)
    elif ext in (".docx",):
        text = get_text_from_docx(content)
    elif ext in (".pptx",):
        text = get_text_from_pptx(content)
    else:
        text = content.decode("utf-8", errors="ignore")

    # Chunk & embed
    chunks = smart_chunk(text)
    df = pd.DataFrame({"chunk_id":[f"{blob_name}#_{i}"] for i in range(len(chunks))})
    df["doc"] = blob_name
    df["text"] = chunks

    embs = embed_texts(chunks)

    # Persist intermediate artifacts to sop-index
    save_parquet("sop-index", f"{blob_name}.chunks.parquet", df)
    # store embeddings as parquet (one file per doc)
    import pyarrow as pa, pyarrow.parquet as pq, io as _io
    table = pa.Table.from_arrays([pa.array(embs.tolist(), type=pa.list_(pa.float32()))], names=["vector"])
    buf = _io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    BS.get_container_client("sop-index").upload_blob(f"{blob_name}.emb.parquet", buf, overwrite=True)

    logging.info(f"[process_upload] Wrote chunks and embeddings for {blob_name}")
