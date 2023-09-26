import os
import argparse
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=str, help='Path to the repository')
    parser.add_argument('--lang', type=str.upper, help='Language of the repository', default='PYTHON')
    args = parser.parse_args()

    repo_path = os.path.normpath(args.repo)
    try:
        lang = getattr(Language, args.lang)
    except AttributeError:
        raise AttributeError(f'Language {args.lang} not found')

    # Load codes
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=lang)
    )
    documents = loader.load()
    print(f'Loaded {len(documents)} documents')

    # Split codes
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang,
        chunk_size=2000,
        chunk_overlap=100
    )
    texts = python_splitter.split_documents(documents)
    print(f'Splitted into {len(texts)} texts')

    # RetrievalQA
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    # Chat
    # Because OpenAI GPT-4 model is involved, we need to set the environment variable OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-4")
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    question = "Could you make a high-level summary of this project with technical details?"
    result = qa(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")