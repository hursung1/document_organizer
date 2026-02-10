from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
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


class ArxivFetcherService:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self.state_path = settings.arxiv_storage_dir / "_state.json"

    def run_once(self, target_dt: datetime | None = None) -> ArxivFetchSummary:
        now_kst = (target_dt.astimezone(KST) if target_dt else datetime.now(KST))
        target_date = now_kst.date() - timedelta(days=1)
        day_token = target_date.strftime("%Y%m%d")

        fetched_total = 0
        by_topic: dict[str, int] = {}
        merged: dict[str, ArxivPaper] = {}
        fetched_unique_ids: set[str] = set()

        for topic in self.settings.arxiv_topics:
            papers = self._fetch_topic(topic=topic, day_token=day_token)
            by_topic[topic] = len(papers)
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
        self._save_state(summary=summary, run_at=now_kst)
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

    def _fetch_topic(self, topic: str, day_token: str) -> list[ArxivPaper]:
        query = f'all:"{topic}" AND submittedDate:[{day_token}0000 TO {day_token}2359]'
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
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_data = response.read()

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
                    matched_topics=[topic],
                )
            )
        return papers

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
        payload = {
            "last_success_date": summary.target_date,
            "last_run_at": run_at.isoformat(),
            "last_summary": summary.to_dict(),
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
