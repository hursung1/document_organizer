# doc-organizer

문서 OCR 적재 + ArXiv 수집 + RAG 채팅을 한 서비스로 제공하는 FastAPI 프로젝트입니다.

## 핵심 기능

- PDF 문서를 GLM-OCR로 파싱하고 Milvus(`doc_chunks`)에 적재
- ArXiv 논문을 매일 오전 9시(KST) 기준으로 갱신
  - 수집 대상: **KST 전날** 제출 논문
  - 키워드별 수집 후 중복 제거
  - JSON 저장 + Milvus(`arxiv_papers`) 적재
- 대화형 RAG API
  - 일반 응답: `POST /api/chat`
  - SSE 스트리밍 응답: `POST /api/chat/stream`
- Redis 기반 대화 이력 저장/조회/삭제
- 새 채팅 스타터 문서 3개 제공
  - ArXiv 결과 우선
  - 카드 설명은 한국어 1문장 요약

## 프로젝트 구조

- `/Users/seonghwan/projects/doc_organizer/main.py`: FastAPI 앱, 스케줄러, API 엔드포인트
- `/Users/seonghwan/projects/doc_organizer/doc_organizer/ingest.py`: OCR 문서 적재 서비스
- `/Users/seonghwan/projects/doc_organizer/doc_organizer/arxiv_fetcher.py`: ArXiv 수집 서비스
- `/Users/seonghwan/projects/doc_organizer/doc_organizer/arxiv_ingest.py`: ArXiv -> Milvus 적재 서비스
- `/Users/seonghwan/projects/doc_organizer/doc_organizer/qa_service.py`: RAG 질의응답 서비스
- `/Users/seonghwan/projects/doc_organizer/doc_organizer/settings.py`: 환경변수/설정 로더

## 요구 사항

- Python 3.12+
- Milvus
- Redis
- OCR 모델 실행 환경(GLM-OCR + `config.yaml`)
- 임베딩 모델(Ollama)
- LLM Provider
  - 기본: Gemini (`LLM_PROVIDER=gemini`)
  - 대안: Ollama (`LLM_PROVIDER=ollama`)

## 빠른 시작

1. 의존성 설치

```bash
uv sync
```

2. 서버 실행

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

3. (선택) OCR 적재 1회 실행

```bash
uv run python model.py
```

## 주요 API

- `GET /health`
  - 런타임 상태, 최근 적재 상태, ArXiv 상태 확인
- `POST /ingest/run`
  - OCR 적재 수동 실행
- `POST /arxiv/run`
  - ArXiv 수집 + Milvus 적재 수동 실행
- `GET /api/starter-docs`
  - 새 채팅 카드용 문서 3개 반환
- `GET /api/conversations`
  - 대화 목록 조회
- `GET /api/conversations/{conversation_id}/messages`
  - 대화 메시지 조회
- `DELETE /api/conversations/{conversation_id}`
  - 대화 삭제
- `POST /api/chat`
  - 일반 질의응답
- `POST /api/chat/stream`
  - SSE 스트리밍 질의응답 (`stage`, `final`, `error`, `done` 이벤트)

## ArXiv 수집/적재 동작

- 실행 시점
  - 매분 체크
  - 현재 시간이 `ARXIV_SCHEDULE_HOUR`(기본 9시, KST) 이후이고,
  - 전날(KST) 수집 성공 기록이 없으면 즉시 실행
- 수집 결과 저장
  - `/Users/seonghwan/projects/doc_organizer/results/arxiv/YYYY-MM-DD.json`
  - `/Users/seonghwan/projects/doc_organizer/results/arxiv/_state.json`
- Milvus 적재 방식
  - 임베딩 입력: `title + summary` 결합 텍스트
  - metadata: `arxiv_id`, `published`, `updated`, `authors`, `categories`, `pdf_url`, `source_url`, `matched_topics`, `ingested_at`
  - 컬렉션: `ARXIV_COLLECTION` (기본 `arxiv_papers`)

## 환경변수

### 공통 경로/입출력

- `DOCS_DIR` (기본: `src_docs`)
- `RESULTS_DIR` (기본: `results`)
- `OCR_CONFIG_PATH` (기본: `config.yaml`)

### Milvus/Ollama

- `MILVUS_COLLECTION` (기본: `doc_chunks`)
- `MILVUS_URI` (기본: `tcp://localhost:19530`)
- `MILVUS_USER`
- `MILVUS_PASSWORD`
- `OLLAMA_HOST` (기본: `http://localhost:11434`)
- `EMBED_MODEL` (기본: `embeddinggemma:latest`)

### OCR 적재

- `CHUNK_SIZE` (기본: `1000`)
- `CHUNK_OVERLAP` (기본: `200`)
- `AUTO_INGEST_ENABLED` (기본: `false`)
- `INGEST_INTERVAL_SECONDS` (기본: `21600`)

### RAG/LLM

- `RETRIEVAL_TOP_K` (기본: `5`)
- `DENSE_WEIGHT` (기본: `0.7`)
- `SPARSE_WEIGHT` (기본: `0.3`)
- `LLM_PROVIDER` (기본: `gemini`, 선택: `gemini` / `ollama`)
- `QA_MODEL` (기본: `qwen3:latest`, Ollama provider에서 사용)
- `GEMINI_MODEL` (기본: `gemini-2.0-flash`)
- `GEMINI_API_KEY` (Gemini provider 사용 시 필수)
- `REDIS_URL` (기본: `redis://localhost:6379/0`)

### ArXiv 수집/적재

- `ARXIV_STORAGE_DIR` (기본: `results/arxiv`)
- `ARXIV_COLLECTION` (기본: `arxiv_papers`)
- `ARXIV_SCHEDULE_HOUR` (기본: `9`)
- `ARXIV_MAX_RESULTS_PER_TOPIC` (기본: `50`)
- `ARXIV_TOPICS` (기본: `computer science,artificial intelligence,large language model,LLM`)

### ArXiv PDF 원문 보강(질의응답 시)

- `ARXIV_PDF_ENRICH_ENABLED` (기본: `true`)
- `ARXIV_PDF_CACHE_DIR` (기본: `results/arxiv/pdf_cache`)
- `ARXIV_PDF_CACHE_TTL_HOURS` (기본: `168`)
- `ARXIV_PDF_MAX_DOCS_PER_QUERY` (기본: `2`)
- `ARXIV_PDF_MAX_SNIPPETS_PER_DOC` (기본: `3`)
- `ARXIV_PDF_MIN_SCORE` (기본: `0.35`)
- `ARXIV_PDF_DOWNLOAD_TIMEOUT_SEC` (기본: `30`)
- `ARXIV_PDF_EXTRACT_MAX_CHARS` (기본: `120000`)

## 참고

- 서비스 시작 시 `AUTO_INGEST_ENABLED=true`인 경우에만 OCR 주기 적재 루프가 동작합니다.
- ArXiv 루프는 항상 시작되며, 실행 조건을 만족할 때만 수집/적재를 수행합니다.
