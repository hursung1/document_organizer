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
- `POST /arxiv/run`: ArXiv 논문 수동 업데이트
- `GET /health`: 상태 확인
- 서비스 시작 후 주기적으로 적재 실행 (`INGEST_INTERVAL_SECONDS`, 기본 21600초=6시간)
- ArXiv 논문은 매일 오전 9시에 자동 업데이트
- 서버가 오전 9시(KST) 이후에 실행되었고 전날(KST) 업데이트 내역이 없으면 즉시 1회 업데이트

### ArXiv 논문 수집

- 수집 대상 기본 키워드: `computer science`, `artificial intelligence`, `large language model`, `LLM`
- 전날(KST) 제출(`submittedDate`) 논문을 주제별로 조회하여 중복 제거 후 저장
- 수집된 각 논문은 Milvus에도 함께 적재
- 임베딩 대상 텍스트: `title + summary` 결합 문자열
- metadata: `arxiv_id`, `published`, `updated`, `authors`, `categories`, `pdf_url`, `source_url`, `matched_topics`
- 저장 위치:
1. 일자별 결과: `results/arxiv/YYYY-MM-DD.json`
2. 상태 파일: `results/arxiv/_state.json`

### 문서 기반 QA(RAG)

- `POST /api/chat`
- 동작 순서:
1. 사용자 질의 분석 후 검색 질의 생성
2. Milvus Hybrid Search (Dense Vector + BM25)로 관련 청크 검색
3. 검색 결과를 근거로 답변 생성

구현 스택:
- LangChain + LangGraph
- LangGraph checkpointer: Redis

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
- `REDIS_URL` (기본: `redis://localhost:6379/0`)
- `ARXIV_STORAGE_DIR` (기본: `results/arxiv`)
- `ARXIV_COLLECTION` (기본: `arxiv_papers`)
- `ARXIV_SCHEDULE_HOUR` (기본: `9`)
- `ARXIV_MAX_RESULTS_PER_TOPIC` (기본: `50`)
- `ARXIV_TOPICS` (쉼표 구분, 기본: `computer science,artificial intelligence,large language model,LLM`)
