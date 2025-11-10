# 은둔/고립 청년 사회복귀 지원 챗봇 API

은둔/고립 청년의 원활한 사회복귀를 돕는 RAG 기반 챗봇 API

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
# venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에 OpenAI API 키 입력
# OPENAI_API_KEY=your_api_key_here
```

## PDF 파일 준비

학습할 PDF 파일들을 `data/` 폴더에 배치하세요.

```bash
# data 폴더 생성 (없는 경우)
mkdir -p data

# PDF 파일 복사 (예시)
cp ~/Downloads/support_guide.pdf data/
cp ~/Downloads/program_info.pdf data/
```

**주의**: 초기화 시 `data/` 폴더의 **모든 PDF 파일**이 자동으로 로드됩니다.

## 실행 방법

```bash
# 서버 실행
python run.py
```

또는

```bash
uvicorn app.main:app --reload
```

## API 사용법

### 1. 챗봇 초기화 (필수)

서버 시작 후 가장 먼저 챗봇을 초기화해야 합니다. `data/` 폴더의 모든 PDF가 자동으로 로드됩니다.

```bash
curl -X POST http://localhost:8000/api/chatbot/initialize
```

응답 예시:
```json
{
  "success": true,
  "message": "챗봇 초기화 완료! 2개의 PDF 파일을 로드했습니다.",
  "data": {
    "document_count": 245,
    "vector_db_path": "./chroma_db",
    "loaded_files": ["support_guide.pdf", "program_info.pdf"]
  }
}
```

### 2. 챗봇 상태 확인

```bash
curl http://localhost:8000/api/chatbot/status
```

### 3. 질문하기

```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "사회복귀를 위한 지원 프로그램에는 어떤 것들이 있나요?"}'
```

예쁘게 출력 (jq 사용):
```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "사회복귀를 위한 지원 프로그램에는 어떤 것들이 있나요?"}' | jq
```

응답 예시:
```json
{
  "success": true,
  "message": "답변이 생성되었습니다.",
  "data": {
    "question": "사회복귀를 위한 지원 프로그램에는 어떤 것들이 있나요?",
    "answer": "은둔/고립 청년을 위한 사회복귀 지원 프로그램은...",
    "sources": ["지원 프로그램 안내...", "상담 서비스 정보..."]
  }
}
```

## API 엔드포인트

### Health Check
- `GET /api/health` - 서버 상태 확인

### Chatbot
- `POST /api/chatbot/initialize` - 챗봇 초기화 (PDF 로드 및 벡터 스토어 생성)
- `GET /api/chatbot/status` - 챗봇 상태 확인
- `POST /api/chatbot/chat` - 질문하기

## API 문서

서버 실행 후 다음 주소에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
h-sw-h-back/
├── app/
│   ├── main.py              # FastAPI 애플리케이션
│   ├── config.py            # 설정
│   ├── routers/             # API 라우터
│   │   ├── health.py
│   │   └── chatbot.py
│   ├── services/            # 비즈니스 로직
│   │   └── vector_store.py
│   ├── utils/               # 유틸리티
│   │   └── pdf_loader.py
│   ├── schemas/             # Pydantic 스키마
│   │   └── chatbot.py
│   └── models/              # 데이터베이스 모델
├── data/
│   └── housing_faq.pdf      # 학습할 PDF 파일
├── chroma_db/               # 벡터 스토어 (자동 생성)
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## 기술 스택

- FastAPI: 웹 프레임워크
- LangChain: RAG 구현
- OpenAI: GPT-3.5-turbo
- ChromaDB: 벡터 스토어
- Sentence Transformers: 임베딩

## 작동 원리

1. PDF 파일을 로드하고 청크로 분할
2. 각 청크를 OpenAI 임베딩으로 변환
3. ChromaDB에 벡터로 저장
4. 사용자 질문을 임베딩으로 변환
5. 가장 유사한 문서 청크 검색
6. GPT-3.5-turbo로 답변 생성
