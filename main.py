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

    # Defined suffixes given the value of lang
    suffixes = {
        Language.CPP: ['.cpp', '.hpp', '.c', '.h'],
        Language.GO: ['.go', '.g'],
        Language.JAVA: ['.java', '.class'],
        Language.JS: ['.js'],
        Language.PHP: ['.php'],
        Language.PROTO: ['.proto'],
        Language.PYTHON: ['.py'],
        Language.RST: ['.rst'],
        Language.RUBY: ['.rb'],
        Language.RUST: ['.rs'],
        Language.SCALA: ['.scala'],
        Language.SWIFT: ['.swift'],
        Language.MARKDOWN: ['.md', '.markdown'],
        Language.LATEX: ['.tex'],
        Language.HTML: ['.html', '.htm'],
        Language.SOL: ['.sol'],
        Language.CSHARP: ['.cs']
    }

    # Load codes
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob='**/[!.]*',
        suffixes=suffixes[lang],
        parser=LanguageParser(language=lang)
    )
    documents = loader.load()
    print(f'Loaded {len(documents)} documents')

    # Split codes
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang,
        chunk_size=4096,
        chunk_overlap=100
    )
    texts = python_splitter.split_documents(documents)
    print(f'Splitted into {len(texts)} texts')

    # RetrievalQA
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = Chroma.from_documents(texts, embedding=embeddings)
    retriever = db.as_retriever(
        search_type="similarity",  # "similarity" (default), "mmr", or "similarity_score_threshold"
        search_kwargs={"k": 16},
    )

    # Chat
    # Because OpenAI GPT-4 model is involved, we need to set the environment variable OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

    question = f"""
    You are an expert on the {lang} programming language.
    Could you show me the detailed technical summary about this project?
    """
    result = qa(question)

    # Print the question and the answer
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")