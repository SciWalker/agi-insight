'''
This code reads in all the text files in the Auto-GPT-Plugins folder and uses the OpenAI API to embed them in a vector space. 
It then uses the pinecone library to store the vectors in an index. 
This code embeds the documents in chunks using a character based text splitter. 
This is necessary because the OpenAI API has a limit on the number of tokens it can process at once. 
This code uses the pinecone library to store the vectors in the pinecone index. 
'''
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
import pinecone
from langchain.text_splitter import CharacterTextSplitter
embeddings = OpenAIEmbeddings(disallowed_special=())

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment='us-east1-gcp'
)


root_dir = '../Auto-GPT-Plugins/'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try: 
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e: 
            pass



text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(texts)
vector_store = Pinecone(
    index=pinecone.Index('pinecone-index'),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace='algorithm'
)
vector_store.from_documents(
    documents=texts, 
    embedding=embeddings,
    index_name='pinecone-index',
    namespace='twitter-algorithm'
)