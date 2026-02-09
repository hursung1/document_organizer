# Document organizer
프로젝트 개요
문서들을 parsing, database에 적재하여 

## GLM-OCR -> Milvus 적재

`model.py`(1회 실행)와 `main.py`(FastAPI 서비스)는 아래 순서로 동작합니다.

1. `src_docs` 디렉토리 생성
2. `src_docs` 내 PDF를 GLM-OCR로 텍스트 추출
3. 텍스트 청크를 Ollama `embeddinggemma:latest`로 임베딩
4. Milvus 컬렉션에 저장

### 실행

```bash
uv sync
python model.py
```

### FastAPI 서비스 실행

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- `POST /ingest/run`: 수동 1회 적재
- `GET /health`: 상태 확인
- 서비스 시작 후 주기적으로 적재 실행 (`INGEST_INTERVAL_SECONDS`, 기본 21600초=6시간)

### 문서 기반 QA(RAG)

- `POST /api/rag/chat`
- 동작 순서:
1. 사용자 질의 분석 후 검색 질의 생성
2. Milvus Hybrid Search (Dense Vector + BM25)로 관련 청크 검색
3. 검색 결과를 근거로 답변 생성

요청 예시:

```json
{
  "message": "품의서에서 결재 절차와 금액 기준을 알려줘"
}
```

### 선택 환경변수

- `MILVUS_URI` (기본: `http://localhost:19530`)
- `MILVUS_USER`
- `MILVUS_PASSWORD`
- `OLLAMA_HOST` (기본: `http://localhost:11434`)
- `EMBED_MODEL` (기본: `embeddinggemma:latest`)
- `QA_MODEL` (기본: `qwen3:latest`)
- `RETRIEVAL_TOP_K` (기본: `5`)
- `DENSE_WEIGHT` (기본: `0.7`)
- `SPARSE_WEIGHT` (기본: `0.3`)
