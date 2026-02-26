from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .settings import IngestionSettings

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
KST = ZoneInfo("Asia/Seoul")


@dataclass(slots=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    updated: str
    authors: list[str]
    categories: list[str]
    pdf_url: str
    source_url: str
    matched_topics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArxivFetchSummary:
    target_date: str
    fetched_count: int
    unique_count: int
    new_count: int
    stored_total: int
    by_topic: dict[str, int]
    output_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArxivBackfillSummary:
    start_date: str
    end_date: str
    days_processed: int
    fetched_count: int
    unique_count: int
    new_count: int
    stored_total: int
    by_topic: dict[str, int]
    output_files: list[str]
    daily_summaries: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivFetcherService:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self.state_path = settings.arxiv_storage_dir / "_state.json"
        self._last_request_monotonic: float | None = None

    def run_once(self, target_dt: datetime | None = None) -> ArxivFetchSummary:
        now_kst = (target_dt.astimezone(KST) if target_dt else datetime.now(KST))
        target_date = now_kst.date() - timedelta(days=1)
        return self._run_for_date(target_date=target_date, run_at=now_kst, save_state=True)

    def run_last_month(
        self,
        target_dt: datetime | None = None,
        lookback_days: int = 30,
    ) -> ArxivBackfillSummary:
        now_kst = (target_dt.astimezone(KST) if target_dt else datetime.now(KST))
        days = max(1, lookback_days)
        end_date = now_kst.date()
        start_date = end_date - timedelta(days=days - 1)

        by_topic_total: dict[str, int] = {
            f"cat:{category}": 0 for category in self.settings.arxiv_categories
        }
        output_files: list[str] = []
        daily_summaries: list[dict[str, Any]] = []
        fetched_count = 0
        unique_count = 0
        new_count = 0
        stored_total = 0

        for offset in range(days):
            target_date = start_date + timedelta(days=offset)
            daily = self._run_for_date(target_date=target_date, run_at=now_kst, save_state=False)
            daily_summaries.append(daily.to_dict())
            output_files.append(daily.output_file)
            fetched_count += daily.fetched_count
            unique_count += daily.unique_count
            new_count += daily.new_count
            stored_total += daily.stored_total
            for topic, count in daily.by_topic.items():
                by_topic_total[topic] = by_topic_total.get(topic, 0) + count

        # Keep scheduler catch-up semantics intact after manual backfill.
        scheduler_target = now_kst.date() - timedelta(days=1)
        last_success_date = (
            scheduler_target
            if start_date <= scheduler_target <= end_date
            else end_date
        )
        summary = ArxivBackfillSummary(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            days_processed=days,
            fetched_count=fetched_count,
            unique_count=unique_count,
            new_count=new_count,
            stored_total=stored_total,
            by_topic=by_topic_total,
            output_files=output_files,
            daily_summaries=daily_summaries,
        )
        self._save_state_payload(
            last_success_date=last_success_date.isoformat(),
            last_summary=summary.to_dict(),
            run_at=now_kst,
        )
        return summary

    def _run_for_date(
        self,
        target_date: date,
        run_at: datetime,
        save_state: bool,
    ) -> ArxivFetchSummary:
        day_token = target_date.strftime("%Y%m%d")

        fetched_total = 0
        by_topic: dict[str, int] = {}
        merged: dict[str, ArxivPaper] = {}
        fetched_unique_ids: set[str] = set()

        for category in self.settings.arxiv_categories:
            label = f"cat:{category}"
            papers = self._fetch_category(category=category, day_token=day_token)
            by_topic[label] = len(papers)
            fetched_total += len(papers)
            for paper in papers:
                fetched_unique_ids.add(paper.arxiv_id)
                existing = merged.get(paper.arxiv_id)
                if existing is None:
                    merged[paper.arxiv_id] = paper
                    continue
                merged_topics = sorted(set(existing.matched_topics + paper.matched_topics))
                existing.matched_topics = merged_topics

        output_path = self._daily_file_path(target_date)
        previous = self._load_daily(output_path)
        previous_ids = set(previous.keys())

        for arxiv_id, paper in previous.items():
            existing = merged.get(arxiv_id)
            if existing is None:
                merged[arxiv_id] = paper
                continue
            existing.matched_topics = sorted(set(existing.matched_topics + paper.matched_topics))

        ordered_papers = sorted(
            (paper.to_dict() for paper in merged.values()),
            key=lambda item: item.get("published", ""),
            reverse=True,
        )
        output_path.write_text(
            json.dumps(ordered_papers, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        new_count = len([paper_id for paper_id in merged if paper_id not in previous_ids])
        summary = ArxivFetchSummary(
            target_date=target_date.isoformat(),
            fetched_count=fetched_total,
            unique_count=len(fetched_unique_ids),
            new_count=new_count,
            stored_total=len(ordered_papers),
            by_topic=by_topic,
            output_file=str(output_path),
        )
        if save_state:
            self._save_state(summary=summary, run_at=run_at)
        return summary

    def should_run(self, now: datetime | None = None) -> bool:
        current_kst = (now.astimezone(KST) if now else datetime.now(KST))
        scheduled = current_kst.replace(
            hour=self.settings.arxiv_schedule_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        if current_kst < scheduled:
            return False
        target_date = current_kst.date() - timedelta(days=1)
        last_success = self.last_success_date()
        return last_success != target_date

    def last_success_date(self) -> date | None:
        state = self._load_state()
        if not state:
            return None
        date_str = state.get("last_success_date")
        if not date_str:
            return None
        try:
            return date.fromisoformat(str(date_str))
        except ValueError:
            return None

    def _fetch_category(self, category: str, day_token: str) -> list[ArxivPaper]:
        query = f"cat:{category} AND submittedDate:[{day_token}0000 TO {day_token}2359]"
        return self._fetch_by_query(
            query=query,
            matched_label=f"cat:{category}",
        )

    def _fetch_by_query(self, query: str, matched_label: str) -> list[ArxivPaper]:
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": 0,
                "max_results": self.settings.arxiv_max_results_per_topic,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API_URL}?{params}"
        xml_data = self._request_with_retry(url)

        root = ET.fromstring(xml_data)
        papers: list[ArxivPaper] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            source_url = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            if not source_url:
                continue
            paper_id = source_url.rsplit("/", 1)[-1]
            title = self._normalize_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
            summary = self._normalize_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
            published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
            updated = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()
            authors = [
                self._normalize_space(author.findtext("atom:name", default="", namespaces=ATOM_NS))
                for author in entry.findall("atom:author", ATOM_NS)
                if self._normalize_space(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            ]
            categories = [
                (category.attrib.get("term", "") or "").strip()
                for category in entry.findall("atom:category", ATOM_NS)
                if (category.attrib.get("term", "") or "").strip()
            ]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            for link in entry.findall("atom:link", ATOM_NS):
                href = (link.attrib.get("href", "") or "").strip()
                if not href:
                    continue
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = href
                    break

            papers.append(
                ArxivPaper(
                    arxiv_id=paper_id,
                    title=title,
                    summary=summary,
                    published=published,
                    updated=updated,
                    authors=authors,
                    categories=categories,
                    pdf_url=pdf_url,
                    source_url=source_url,
                    matched_topics=[matched_label],
                )
            )
        return papers

    def _request_with_retry(self, url: str) -> bytes:
        max_attempts = max(1, self.settings.arxiv_retry_max_attempts)
        base_delay = max(0.1, self.settings.arxiv_retry_base_delay_sec)
        timeout = max(1, self.settings.arxiv_request_timeout_sec)

        for attempt in range(max_attempts):
            self._respect_min_interval()
            request = urllib.request.Request(
                url=url,
                headers={
                    "User-Agent": self.settings.arxiv_user_agent,
                    "Accept": "application/atom+xml",
                },
            )
            self._last_request_monotonic = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                status_code = int(exc.code)
                retryable = status_code in {429, 500, 502, 503, 504}
                if not retryable or attempt >= max_attempts - 1:
                    raise
                delay = self._retry_delay_from_response(exc, base_delay, attempt)
                time.sleep(delay)
            except urllib.error.URLError:
                if attempt >= max_attempts - 1:
                    raise
                delay = base_delay * (2**attempt)
                time.sleep(delay)

        raise RuntimeError("ArXiv API request failed after retries.")

    def _respect_min_interval(self) -> None:
        min_interval = max(0.0, self.settings.arxiv_min_request_interval_sec)
        if min_interval <= 0:
            return
        if self._last_request_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    @staticmethod
    def _retry_delay_from_response(
        exc: urllib.error.HTTPError,
        base_delay: float,
        attempt: int,
    ) -> float:
        retry_after = (exc.headers.get("Retry-After") or "").strip()
        if retry_after:
            if retry_after.isdigit():
                return max(base_delay, float(retry_after))
            try:
                retry_dt = parsedate_to_datetime(retry_after)
                now_dt = datetime.now(retry_dt.tzinfo) if retry_dt.tzinfo else datetime.utcnow()
                delta = (retry_dt - now_dt).total_seconds()
                if delta > 0:
                    return max(base_delay, delta)
            except Exception:
                pass
        return base_delay * (2**attempt)

    @staticmethod
    def _normalize_space(text: str | None) -> str:
        if text is None:
            return ""
        return " ".join(text.split())

    def _daily_file_path(self, target_date: date) -> Path:
        return self.settings.arxiv_storage_dir / f"{target_date.isoformat()}.json"

    def _load_daily(self, path: Path) -> dict[str, ArxivPaper]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(raw, list):
            return {}

        loaded: dict[str, ArxivPaper] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            arxiv_id = str(item.get("arxiv_id", "")).strip()
            if not arxiv_id:
                continue
            loaded[arxiv_id] = ArxivPaper(
                arxiv_id=arxiv_id,
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                published=str(item.get("published", "")),
                updated=str(item.get("updated", "")),
                authors=[str(author) for author in item.get("authors", []) if str(author).strip()],
                categories=[
                    str(category) for category in item.get("categories", []) if str(category).strip()
                ],
                pdf_url=str(item.get("pdf_url", "")),
                source_url=str(item.get("source_url", "")),
                matched_topics=[
                    str(topic) for topic in item.get("matched_topics", []) if str(topic).strip()
                ],
            )
        return loaded

    def _save_state(self, summary: ArxivFetchSummary, run_at: datetime) -> None:
        self._save_state_payload(
            last_success_date=summary.target_date,
            last_summary=summary.to_dict(),
            run_at=run_at,
        )

    def _save_state_payload(
        self,
        last_success_date: str,
        last_summary: dict[str, Any],
        run_at: datetime,
    ) -> None:
        payload = {
            "last_success_date": last_success_date,
            "last_run_at": run_at.isoformat(),
            "last_summary": last_summary,
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data
