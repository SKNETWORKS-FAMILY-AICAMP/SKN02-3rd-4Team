import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain



# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정 함수
def get_environment_variable(key):
    if key in os.environ:
        value = os.environ.get(key)
    else:
        value = input(f"Insert your {key}: ")
    return value

# OpenAI API 키 가져오기
openai_api_key = get_environment_variable("OPENAI_API_KEY")

# 벡터 스토어 로드 함수
def load_vectorstore(vectorstore_name):
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_api_key)
    vector_store = Chroma(
        persist_directory=f"./data/vector_stores/{vectorstore_name}",
        embedding_function=embeddings
    )
    return vector_store

# 리트리버 생성 함수
def create_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )
    return retriever

# 메모리 생성 함수
def create_memory():
    memory = ConversationSummaryBufferMemory(
        max_token_limit=1024,
        llm=ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0.1),
        return_messages=True,
        memory_key='chat_history',
        output_key="answer",
        input_key="question"
    )
    return memory

# LLM 인스턴스 생성 함수
def instantiate_LLM(model_name="gpt-3.5-turbo", temperature=0.5):
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model=model_name,
        temperature=temperature
    )
    return llm

# Conversational Retrieval Chain 생성 함수
def create_conversational_chain(llm, retriever):
    standalone_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        template="""다음의 대화 내용과 후속 질문을 참고하여, 후속 질문을 독립된 질문으로 재구성해 주세요.
                    질문은 원래 언어(한국어)로 작성되어야 합니다.

                    대화 기록:
                    {chat_history}

                    후속 질문: {question}

                    독립된 질문:"""
    )

    answer_prompt = ChatPromptTemplate.from_template(
                    """다음의 대화 기록과 문서 내용을 바탕으로, 독립된 질문에 대한 답변을 생성해 주세요. 
                        질문: {question}
                        대화 기록: 
                        {chat_history}
                        문서 내용:
                        {context}

                        답변:"""
    )

    memory = create_memory()

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=standalone_question_prompt,
        combine_docs_chain_kwargs={'prompt': answer_prompt},
        condense_question_llm=llm,
        memory=memory,
        retriever=retriever,
        llm=llm,
        chain_type="stuff",
        verbose=False,
        return_source_documents=True    
    )
    
    return chain

# 메인 함수
def get_response(question):
    vector_store_name = "Korean_PDF_OpenAI_Embeddings"
    
    # 벡터 스토어 로드
    vector_store = load_vectorstore(vector_store_name)

    # 리트리버 생성
    retriever = create_retriever(vector_store)

    # LLM 생성
    llm = instantiate_LLM()

    # Conversational Retrieval Chain 생성
    chain = create_conversational_chain(llm, retriever)

    response = chain.invoke({"question": question})
    answer = response['answer']

    return answer