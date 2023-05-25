from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.vectorstores import Pinecone
import pinecone
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_organization=os.environ['OPENAI_ORG_ID'],
)
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment='us-east1-gcp'
)
db = Pinecone(
    index=pinecone.Index("pinecone-index"),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace='twitter-algorithm'
)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

def filter(x):
    # filter based on source code
    if 'com.google' in x['text'].data()['value']:
        return False
    
    # filter based on path e.g. extension
    metadata =  x['metadata'].data()['value']
    return 'scala' in metadata['source'] or 'py' in metadata['source']

### turn on below for custom filtering
# retriever.search_kwargs['filter'] = filter


model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
def main():
    chat_history = []

    while True:
        question = input("Enter a question (or type 'quit' to stop): ")
        if question.lower() == 'quit':
            break

        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
        #save chat history
        with open('chat_history.txt', 'w') as f:
            for item in chat_history:
                f.write("%s\n" % str(item))

main()
