# FastAPI Backend

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

## 실행 방법

```bash
# .env 파일 생성
cp .env.example .env

# 서버 실행
python run.py
```

또는

```bash
uvicorn app.main:app --reload
```

## API 문서

서버 실행 후 다음 주소에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
h-sw-h-back/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 애플리케이션
│   ├── config.py        # 설정
│   ├── routers/         # API 라우터
│   ├── models/          # 데이터베이스 모델
│   └── schemas/         # Pydantic 스키마
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```
