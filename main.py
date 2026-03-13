import pathway as pw
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm import llms
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.retrievers import GeminiEmbedder
from pathway.xpacks.llm.parsers import UnstructuredParser , DoclingParser
from pathway.xpacks.llm.splitters import TokenCountSplitter
from dotenv import load_dotenv
import os

load_dotenv()


documents = pw.io.fs.read(path="./files/", format="binary", with_metadata=True , mode="streaming")


text_splitter = TokenCountSplitter(
    min_tokens=100, max_tokens=500, encoding_name="cl100k_base"         # this encoder is for openai , but dosent make a big difference with gemini
)

parser= DoclingParser()

embedder = GeminiEmbedder(api_key=os.getenv("GOOGLE_API_KEY"))

document_store = DocumentStore(
    docs=documents,
    retriever_factory=retriever_factory,
    parser=parser,
    splitter=text_splitter,
)
