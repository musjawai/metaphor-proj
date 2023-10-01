from metaphor_python import Metaphor
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from templates import template_factory
from tqdm import tqdm
from constants import *
import os, openai, datetime

load_dotenv()
metaphor = Metaphor(os.environ.get("METAPHOR_API_KEY"))
openai.api_key = os.environ.get("OPENAI_API_KEY")

class X:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.process_db()
        self.db = self.embed_summaries()

    def generate(self, input, template_id):
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = LLMChain(
            llm=llm, 
            prompt=PromptTemplate.from_template(template=template_factory(template_id))
        )
        generation = chain(input)['text']
        return generation
    
    def get_searches(self, query, days_back=30):
        date = str(datetime.date.today() - datetime.timedelta(days=days_back))
        searches = metaphor.search(
            query, 
            use_autoprompt=True, 
            start_published_date=date,
            num_results=5
            )
        return searches
    
    def store_summary(self, summary):
        ticker_file = os.path.join(ROOT_DIR, f"{self.ticker}.txt")
        if os.path.exists(ticker_file):
            with open(ticker_file, "r") as file:
                existing_content = file.read()
        else:
            existing_content = ""

        updated = existing_content + "\n" + summary
        with open(ticker_file, "w") as file:
            file.write(updated)
    
    def embed_summaries(self):
        raw_docs = TextLoader(f"summaries/{self.ticker}.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_docs)
        db = FAISS.from_documents(documents, OpenAIEmbeddings())
        return db
    
    def process_db(self):
        search_query = self.generate(
            f"What are recent news about the stock with ticker {self.ticker}", 
            'generate_queries'
            )
        results = self.get_searches(search_query, days_back=90)
        for content in tqdm(results.get_contents().contents, desc="Running Experiment"):
            extracted = content.extract
            summary = self.generate(extracted, "summarize_content_from_url")
            self.store_summary(summary)

    def __call__(self, user_input):
        query = self.generate(user_input, "generate_queries")
        docs = self.db.similarity_search(query)[0].page_content
        summary = self.generate(docs, "summarize_content_from_db")
        return summary
    
if __name__ == "__main__":
    obj = X("LEN")
    print(obj("What did management say about future outlook?"))

