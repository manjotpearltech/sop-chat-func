
import logging, io, os, glob
import pandas as pd
import numpy as np
import azure.functions as func
from utils import BS, build_and_store_faiss

def main(myevent: func.TimerRequest) -> None:
    logging.info("[rebuild_index] Starting rebuild")

    cc = BS.get_container_client("sop-index")
    # list all per-doc parquet files and stack them
    chunks = []
    embs = []
    for blob in cc.list_blobs():
        if blob.name.endswith(".chunks.parquet"):
            with io.BytesIO(cc.download_blob(blob.name).readall()) as b:
                chunks.append(pd.read_parquet(b))
        elif blob.name.endswith(".emb.parquet"):
            with io.BytesIO(cc.download_blob(blob.name).readall()) as b:
                import pyarrow.parquet as pq
                t = pq.read_table(b)
                arr = np.array([np.array(x) for x in t["vector"].to_pylist()], dtype=np.float32)
                embs.append(arr)

    if not chunks or not embs:
        logging.warning("[rebuild_index] Nothing to index yet")
        return

    df = pd.concat(chunks, ignore_index=True)
    V = np.vstack(embs)
    build_and_store_faiss(df, V)
    logging.info(f"[rebuild_index] Rebuilt index with {len(df)} chunks")
