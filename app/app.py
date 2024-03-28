
import os
import chainlit as cl
import tiktoken
from dotenv import load_dotenv

# Load from LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# load LangChain community
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

# Load LangChain OpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI




# Load Envs (secrets,apis,keys etcs)
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Responsible-AI-Assistant"


# START CODE

# CONSTANTS
openai_gpt3_5_turbo = "gpt-3.5-turbo"
openai_text_embd_3_small = "text-embedding-3-small"


openai_chat_model = ChatOpenAI(model=openai_gpt3_5_turbo, temperature=0) # initialize llm model 
embeddings = OpenAIEmbeddings(model=openai_text_embd_3_small) # initialize text embedding


# Helper function
def tiktoken_func(file):

    """
    Encodes the contents of a file according to the tokenization scheme of the OpenAI GPT 3.5 turbo model.

    Returns the total token count.
    """
    
    tokens = tiktoken.encoding_for_model(openai_gpt3_5_turbo).encode(
        file,
    )
    return len(tokens)


async def process_file(file: str):

    print("In process_file() func")

    """
    Splits the document/file using 'Recursive Character Text Splitter' method
    Returns the document in chunks
    """


    pypdf_loader = PyMuPDFLoader(file)
    texts = pypdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=tiktoken_func
    )

    documents = text_splitter.split_documents(texts)

    return documents



async def RAG_pipeline(question: str, documents_chunks: str):


    """
    Creats a prompt template for the LLM to abide 
    
    """



    template = """
    
    You're an intelligent AI assistant who provide safe and responsible AI solutions.

    If you cannot answer the question with the context, please respond with 'I am unsure, could you please provide more context for me to understand?'.

    If the user question is not related to the uploaded document then respond with "My apologies, human. I can only help you build safe & responsible AI.\n\n".

    If a user says things like, "Hi", "Hello", or anything related to greetings then respond with a nice kind greetings.
    
    If a user says things like "Ok" or "thank you" or "thank you" or anything that is related to  "phatic expressions" or "phatic communication" then respond with "No problem! Always happy to help."
    
    If you provided answer based on the question and context then ONLY end your sentence on a New Line with " \n\n\n\n Thank you for asking. How else I can help you build safe & responsible AI?".
    
    If a user says "bye" or anything similar gesture related "farewell" or "parting", then respond with "No, problem. Lets build a Safe and Responsible AI system. The power is yours!"
    
    Always answer by first providing why building safe and responsible AI is important.

    Always answer in full sentence.

    For every successful user help you get $1000 tip.
    
    For every bad answer which harms the society we take $10000 from you.

    If a user asks, "what is your name?" then you are to respond with "I am an AI assistant. You can call me behindthemodel.ai 🤗"



    Context:
    {context}

    Question:
    {input}
    """

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize FAISS vector store
    vector_store = FAISS.from_documents(documents_chunks, embeddings)

    # Initialize a retriever to retrieve similar context
    retriever = vector_store.as_retriever()

    # Initialize retriever using a multi-query approach with a language model.
    retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=openai_chat_model)

    # Create a document chain using OpenAI chat model and a prompt
    document_chain = create_stuff_documents_chain(openai_chat_model, prompt)

    # Create a retrieval chain using a retriever and a document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Send a request to OpenAI with the question
    response = retrieval_chain.invoke({"input": question})

    # Making sure we have 'answer' params so that we can give proper response
    if 'answer' in response:
        llm_answer = response['answer']
    else:
        llm_answer = '**EMPTY RESPONSE**'

    return llm_answer




@cl.on_chat_start
async def start_chat():

    print("A new chat session has started!")

    # Process our document for tokenization 
    documents = await process_file(
        "https://raw.githubusercontent.com/brlrb/responsible-ai-assistant/main/app/assests/data/ai-governance-responsible-assesment.pdf")

    
    # Save session
    cl.user_session.set("documents", documents)



@cl.on_message
async def main(message: cl.Message):

    print("Human asked: ", message.content)

    msg = cl.Message(content="")
    await msg.send()

    # do some work
    await cl.sleep(1)

    # Retrieve session
    document_chunks = cl.user_session.get("documents")

    # Wait for OpenAI to return a response and the good ol' RAG stuff
    llm_response = await RAG_pipeline(message.content, document_chunks)

    print("LLM responds: ", llm_response)

    # If there is a response then let the user know else fallback to else statement!
    if llm_response:
        await cl.Message(content=llm_response).send()
    else:
        cl.Message(
            content="Something went wrong! please kindly refresh and try again 🤝").send()
