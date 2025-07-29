import asyncio
import os
import ssl
from typing import List, Dict, Any

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documentation_helper.logger import log_header, log_info, Colors, log_success, log_error, log_warning

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

"""
Ingestion Pipeline:
1. Website Discovery (TavilyMap)
   |- Input: https://python.langchain.com/
   |- Output: List of 200+ documentation URLs
2. URL Batching
   |- Input: 200+ URLs
   |- Output: 10+ batches of 20 URLs each
3. Content Extraction (TavilyExtract)
   |- Input: Batches of URLs
   |- Process: Concurrent extraction from web pages
   |- Output: Clean, parsed content
"""

load_dotenv()

START_URL = "https://python.langchain.com/"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_extract = TavilyExtract()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specific size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        log_info(f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get("results", []))} documents")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []


async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENTATION EXTRACTION")
    log_info(f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches", Colors.DARKCYAN)

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # filter out exceptions
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                all_pages.append(Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]}
                ))
    if failed_batches > 0:
        log_error(f"TavilyExtract: Failed to extract {failed_batches} batches")
    else:
        log_success(f"TavilyExtract: extracted {len(all_pages)} documents")
    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int=50):
    log_header("VECTOR STORAGE")
    log_info(f"VectorStore indexing: Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    # process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"VectorStore: Successfully added batch {batch_num}/{len(batches)} of {len(batch)} documents")
        except Exception as e:
            log_error(f"VectorStore: Failed to add batch {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(f"VectorStore: Indexing successful")
    else:
        log_warning("VectorStore: Some indexing has failed")


async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # log_info(f"TavilyCraw: Starting to Crawl documentation from {START_URL}", Colors.PURPLE)
    #
    # # Craw the documentation site
    # res = tavily_crawl.invoke({
    #     "url": START_URL,
    #     "max_depth": 5,
    #     "extract_depth": "advanced", # retrieve more data like table and embedded content
    #     # "instructions": "natural language defining filtering criteria like content on ai agents"
    # })
    #
    # all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results']]
    # log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} pages")

    # Website Discovery
    log_info(f"TavilyMap: starting to map documentation structure from {START_URL}", Colors.PURPLE)
    site_map = tavily_map.invoke(START_URL)
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site")

    # URL Batching
    url_batches = chunk_urls(list(site_map['results']))
    log_info(f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches", Colors.BLUE)

    # Content Extraction
    all_docs = await async_extract(url_batches)

    # chunking
    log_header("DOCUMENT CHUNKING")
    log_info(f"Text splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # indexing - commenting out to prevent duplicate upload
    # await index_documents_async(splitted_docs, batch_size=500)


if __name__ == "__main__":
    asyncio.run(main())
