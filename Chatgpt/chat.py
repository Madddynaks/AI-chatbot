
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

embeddings = OpenAIEmbeddings()

loader = TextLoader('data.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vecstore = Chroma.from_documents(texts,embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever = vecstore.as_retriever()
)

def query(q):
    print("Query: ",q)
    print("Answer:",qa.run(q))

# query("What is the data about ?")
# query("News about Maharashtra?")
# query("Who is the defence minister ?")
query("Today's cricket World cup match ?")