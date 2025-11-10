from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from typing import Optional
import os

class VectorStoreService:
    """
    RAG 기반 챗봇 서비스
    - PDF 문서를 벡터 DB에 저장
    - 사용자 질문에 대해 문서 기반 답변 제공
    """

    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        """
        초기화 함수
        - OpenAI API 키 설정
        - 임베딩 모델 설정 (text-embedding-3-small)
        """
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory

        # 직접 환경변수 등록 (가장 확실한 방법) - 코랩과 동일
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # 임베딩 모델 설정 - 코랩과 동일
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        self.vectorstore: Optional[Chroma] = None  # 벡터 DB
        self.qa_chain: Optional[RetrievalQA] = None  # 질의응답 체인

    def create_vectorstore(self, documents, batch_size: int = 100):
        """
        벡터 스토어 생성 (배치 처리)
        - 문서들을 임베딩으로 변환
        - ChromaDB에 저장
        - 대용량 문서 처리를 위해 배치 단위로 처리
        """
        print(f"총 {len(documents)}개의 문서 청크를 임베딩합니다...")

        # 배치 단위로 나누어 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"배치 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} 처리 중... ({len(batch)}개)")

            if i == 0:
                # 첫 번째 배치: 새 벡터스토어 생성
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                # 이후 배치: 기존 벡터스토어에 추가
                self.vectorstore.add_documents(batch)

        print("임베딩 완료!")
        return self.vectorstore

    def load_vectorstore(self):
        """
        기존 벡터 스토어 불러오기
        - 이미 저장된 벡터 DB가 있으면 재사용
        """
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vectorstore
        return None

    def create_qa_chain(self, model_name: str = "gpt-4o-mini"):
        """
        QA 체인 생성
        - LLM 모델 설정 (코랩: ChatOpenAI(model='gpt-4o-mini'))
        - 문서 검색 + 답변 생성 체인 구축
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 없습니다. create_vectorstore를 먼저 호출하세요.")

        # LLM 모델 설정 - 코랩처럼 환경변수에서 자동으로 API 키 가져옴
        llm = ChatOpenAI(
            model=model_name,
            temperature=0  # 일관된 답변을 위해 0으로 설정
        )

        # 코랩 코드처럼 친절한 한국어 프롬프트 추가
        from langchain_core.prompts import PromptTemplate

        prompt_template = """너는 친절한 한국어 비서야.
        아래 문서 내용을 참고해서 쉽고 자세하게 질문에 답해줘.
        답변할 때는 존댓말을 사용하고, 핵심 내용을 먼저 말한 후 자세한 설명을 해줘.

        문서 내용:
        {context}

        질문: {question}

        답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # QA 체인 생성 - 문서 검색 + LLM 답변 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 검색된 문서를 모두 프롬프트에 포함
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),  # 상위 3개 문서 검색
            return_source_documents=True,  # 참조 문서도 함께 반환
            chain_type_kwargs={"prompt": PROMPT}  # 커스텀 프롬프트 적용
        )
        return self.qa_chain

    def query(self, question: str) -> dict:
        """
        질문에 답변하기
        코랩 코드:
        hits = db.similarity_search(q, k=3)
        context = "\\n\\n".join(h.page_content for h in hits)
        answer = chain.invoke({'context': context, 'q': q})
        """
        if not self.qa_chain:
            raise ValueError("QA 체인이 없습니다. create_qa_chain을 먼저 호출하세요.")

        # 질문 실행 - 자동으로 문서 검색 + 답변 생성
        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],  # AI 답변
            "sources": [doc.page_content for doc in result["source_documents"]]  # 참조 문서
        }
