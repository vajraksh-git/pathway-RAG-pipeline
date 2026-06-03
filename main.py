import pathway as pw
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm import llms , prompts
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import OpenAIEmbedder , LiteLLMEmbedder

from pathway.xpacks.llm.parsers import UnstructuredParser , DoclingParser
from pathway.xpacks.llm.splitters import TokenCountSplitter
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api=os.getenv("GEMINI_KEY")
genai.configure(api_key=api)

model = genai.GenerativeModel('gemini-2.0-flash')


documents = pw.io.fs.read(path="./files/", format="binary", with_metadata=True , mode="streaming")


text_splitter = TokenCountSplitter(
    min_tokens=100, max_tokens=500, encoding_name="cl100k_base"         # this encoder is for openai , but dosent make a big difference with gemini
)

parser= DoclingParser()

embedder = pw.xpacks.llm.embedders.GeminiEmbedder(
    model="models/gemini-embedding-001",  # Updated model name
    api_key=os.getenv("GEMINI_KEY")
)
retriever_factory = BruteForceKnnFactory(
    embedder=embedder,
)

document_store = DocumentStore(
    docs=documents,
    retriever_factory=retriever_factory,
    parser=parser,
    splitter=text_splitter,
)

webserver = pw.io.http.PathwayWebserver(host="0.0.0.0", port=8011)

class QuerySchema(pw.Schema):
    messages: str


queries, writer = pw.io.http.rest_connector(
    webserver=webserver,
    schema=QuerySchema,
    autocommit_duration_ms=50,
    delete_completed_queries=False,
)

queries = queries.select(
    query = pw.this.messages,
    k = 1,
    metadata_filter = None,
    filepath_globpattern = None,
)


retrieved_documents = document_store.retrieve_query(queries)
retrieved_documents = retrieved_documents.select(docs=pw.this.result)
queries_context = queries + retrieved_documents

def get_context(documents):
    content_list = []
    for doc in documents:
        content_list.append(str(doc["text"]))
    return " ".join(content_list)

@pw.udf
def build_prompts_udf(documents, query) -> str:
    context = get_context(documents)
    prompt = (
        f"Given the following documents : \n {context} \nanswer this query: {query}"
    )
    return prompt


prompts = queries_context+queries_context.select(
    prompts=build_prompts_udf(pw.this.docs, pw.this.query)
)


model = llms.LiteLLMChat(
    model="groq/llama-3.3-70b-versatile", # High-performance free model
    api_key=os.getenv("GROQ_KEY"),
    num_retries=1,
    )

response = prompts.select(
    *pw.this.without(pw.this.query, pw.this.prompts, pw.this.docs),
    result=model(
        llms.prompt_chat_single_qa(pw.this.prompts),
    ),
)

writer(response)
pw.run()
