# app/api_client.py
"""
HTTP client for SmartCampus V2T backend.

Purpose:
- Hide REST details from Streamlit UI
- Provide stable return formats for UI rendering
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = (base_url or "").rstrip("/")

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def health(self) -> Dict[str, Any]:
        try:
            r = requests.get(self._url("/healthz"), timeout=5)
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc)}
        if r.status_code != 200:
            return {"ok": False, "status_code": r.status_code, "text": r.text}
        try:
            return r.json()
        except ValueError:
            return {"ok": False, "status_code": r.status_code, "text": r.text}

    def list_videos(self) -> List[Dict[str, Any]]:
        r = requests.get(self._url("/v1/videos"), timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_video(self, filename: str, data: bytes) -> Dict[str, Any]:
        files = {"file": (filename, data)}
        r = requests.post(self._url("/v1/videos/upload"), files=files, timeout=120)
        r.raise_for_status()
        return r.json()

    def delete_video(self, video_id: str) -> Dict[str, Any]:
        r = requests.delete(self._url(f"/v1/videos/{video_id}"), timeout=60)
        r.raise_for_status()
        return r.json()

    def get_video_outputs(self, video_id: str, lang: str, variant: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"lang": lang}
        if variant:
            params["variant"] = variant
        r = requests.get(self._url(f"/v1/videos/{video_id}/outputs"), params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_batch_manifest(self, video_id: str) -> Dict[str, Any]:
        r = requests.get(self._url(f"/v1/videos/{video_id}/batch-manifest"), timeout=60)
        r.raise_for_status()
        return r.json()

    def get_metrics_summary(self, video_id: str, lang: str, variant: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"lang": lang}
        if variant:
            params["variant"] = variant
        r = requests.get(self._url(f"/v1/videos/{video_id}/metrics-summary"), params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def create_job(
        self,
        video_id: str,
        extra: Optional[Dict[str, Any]] = None,
        profile: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "video_id": video_id,
            "profile": profile,
            "variant": variant,
            "extra": extra or {},
        }
        r = requests.post(self._url("/v1/jobs"), json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        r = requests.get(self._url(f"/v1/jobs/{job_id}"), timeout=30)
        r.raise_for_status()
        return r.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        r = requests.post(self._url(f"/v1/jobs/{job_id}/cancel"), timeout=30)
        r.raise_for_status()
        return r.json()

    def queue_list(self) -> Dict[str, Any]:
        r = requests.get(self._url("/v1/queue"), timeout=30)
        r.raise_for_status()
        return r.json()

    def queue_pause(self) -> Dict[str, Any]:
        r = requests.post(self._url("/v1/queue/pause"), timeout=30)
        r.raise_for_status()
        return r.json()

    def queue_resume(self) -> Dict[str, Any]:
        r = requests.post(self._url("/v1/queue/resume"), timeout=30)
        r.raise_for_status()
        return r.json()

    def queue_remove(self, job_id: str) -> Dict[str, Any]:
        r = requests.delete(self._url(f"/v1/queue/{job_id}"), timeout=30)
        r.raise_for_status()
        return r.json()

    def queue_move(self, job_id: str, direction: str, steps: int = 1) -> Dict[str, Any]:
        payload = {"job_id": str(job_id), "direction": str(direction), "steps": int(steps)}
        r = requests.post(self._url("/v1/queue/move"), json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def index_status(self) -> Dict[str, Any]:
        r = requests.get(self._url("/v1/index/status"), timeout=30)
        r.raise_for_status()
        return r.json()

    def index_rebuild(self) -> Dict[str, Any]:
        r = requests.post(self._url("/v1/index/rebuild"), timeout=600)
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        top_k: int,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        variant: Optional[str] = None,
        dedupe: bool = True,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        event_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        tags: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        people_count_bucket: Optional[str] = None,
        motion_type: Optional[str] = None,
        anomaly_only: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "top_k": int(top_k),
            "video_id": video_id,
            "language": language,
            "variant": variant,
            "dedupe": bool(dedupe),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "min_duration_sec": min_duration_sec,
            "max_duration_sec": max_duration_sec,
            "event_type": event_type,
            "risk_level": risk_level,
            "tags": tags,
            "objects": objects,
            "people_count_bucket": people_count_bucket,
            "motion_type": motion_type,
            "anomaly_only": bool(anomaly_only),
        }
        r = requests.post(self._url("/v1/search"), json=payload, timeout=120)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and "hits" in js and isinstance(js["hits"], list):
            return js["hits"]
        return []

    def build_report(
        self,
        *,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        variant: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        payload = {
            "video_id": video_id,
            "language": language,
            "variant": variant,
            "query": query,
            "top_k": int(top_k),
        }
        r = requests.post(self._url("/v1/reports"), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def ask_qa(
        self,
        *,
        question: str,
        language: Optional[str] = None,
        variant: Optional[str] = None,
        video_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        payload = {
            "question": question,
            "language": language,
            "variant": variant,
            "video_id": video_id,
            "top_k": int(top_k),
        }
        r = requests.post(self._url("/v1/qa"), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def ask_rag(
        self,
        *,
        query: str,
        language: Optional[str] = None,
        variant: Optional[str] = None,
        video_id: Optional[str] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "language": language,
            "variant": variant,
            "video_id": video_id,
            "top_k": int(top_k),
        }
        r = requests.post(self._url("/v1/rag"), json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
