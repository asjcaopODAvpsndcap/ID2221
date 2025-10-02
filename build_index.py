import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
import os
import s3fs  # Requires s3fs library

from dotenv import load_dotenv # 导入 load_dotenv

# --- 1. Key and Region Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# Replace with your actual Secret Access Key
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# Replace with your S3 bucket's region, e.g. 'us-east-1'
AWS_REGION = os.getenv("AWS_REGION")

# --- 2. Other Parameters ---
S3_BUCKET = "id2221"
S3_PREFIX = "10_01/domain_partitioned/"
CHROMA_PERSIST_DIR = "./chroma_db_ollama"
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:latest"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    print(f"--- Starting vector index construction (using Ollama local model: {OLLAMA_EMBEDDING_MODEL}) ---")

    try:
        print(f"Step 1/5: Connecting to S3 and finding all Parquet files...")

        fs = s3fs.S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
            client_kwargs={'region_name': AWS_REGION}
        )

        full_s3_path = f"{S3_BUCKET}/{S3_PREFIX}"

        #  Critical modification: Using fs.find() instead of fs.glob()
        print(f"Recursively searching path using fs.find(): s3://{full_s3_path}")
        # fs.find() returns a dictionary where keys are paths and values are file information
        all_files_info = fs.find(full_s3_path)

        # Filter out files ending with .parquet from all found files
        file_paths = [path for path in all_files_info if path.endswith('.parquet')]

        if not file_paths:
            print(f" Error: Still no .parquet files found in path s3://{full_s3_path}.")
            print("Please check:")
            print("1. Whether S3_BUCKET and S3_PREFIX variables are completely correct.")
            print("2. Whether your IAM key has `s3:ListBucket` permission.")
            return

        print(f"Found {len(file_paths)} Parquet files.")

        all_dfs = []
        for file_path in tqdm(file_paths, desc="Reading Parquet files"):
            s3a_path = f"s3a://{file_path}"
            try:
                storage_options = {
                    "key": AWS_ACCESS_KEY_ID,
                    "secret": AWS_SECRET_ACCESS_KEY
                }
                temp_df = pd.read_parquet(s3a_path, engine='pyarrow', storage_options=storage_options)
                all_dfs.append(temp_df)
            except Exception as e:
                print(f"\n⚠ Warning: Skipping file {s3a_path} because it could not be read. Error: {e}")

        if not all_dfs:
            print(" Error: All found Parquet files could not be read successfully.")
            return

        print("\nStep 2/5: Merging all data...")
        df = pd.concat(all_dfs, ignore_index=True)

        df = df.dropna(subset=['content'])
        df = df[df['content'].str.strip() != '']
        print(f"Successfully loaded and merged {len(df)} valid document records.")

    except Exception as e:
        print(f" Fatal error occurred during data loading and merging: {e}")
        return

    # --- Subsequent code remains unchanged ---
    print("Step 3/5: Splitting document content into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Splitting documents"):
        metadata = {
            "pmid": str(row.get("pmid", "N/A")),
            "title": str(row.get("title", "N/A")),
            "journal": str(row.get("journal", "N/A")),
            "year": str(row.get("year", "N/A")),
            "source": str(row.get("ref_id", "N/A"))
        }
        content_text = str(row["content"]) if pd.notna(row["content"]) else ""
        chunks = text_splitter.split_text(content_text)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=metadata))

    print(f"All documents successfully split into {len(all_chunks)} text chunks.")

    print(f"Step 4/5: Loading Ollama embedding model '{OLLAMA_EMBEDDING_MODEL}'...")
    embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    print("Step 5/5: Storing all text chunks and their vectors in local ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print("\n--- Vector index construction complete! ---")
    print(f"Vector database successfully saved to '{CHROMA_PERSIST_DIR}' folder.")

