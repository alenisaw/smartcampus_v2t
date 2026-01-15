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
        r = requests.get(self._url("/healthz"), timeout=5)
        if r.status_code != 200:
            return {"ok": False, "status_code": r.status_code, "text": r.text}
        return r.json()

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

    def list_runs(self) -> Dict[str, List[str]]:
        r = requests.get(self._url("/v1/runs"), timeout=30)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and "runs_map" in js and isinstance(js["runs_map"], dict):
            return js["runs_map"]
        if isinstance(js, dict):
            return js
        return {}

    def get_run(self, video_id: str, run_id: str) -> Dict[str, Any]:
        r = requests.get(self._url(f"/v1/runs/{video_id}/{run_id}"), timeout=60)
        r.raise_for_status()
        return r.json()

    def delete_run(self, video_id: str, run_id: str) -> Dict[str, Any]:
        r = requests.delete(self._url(f"/v1/runs/{video_id}/{run_id}"), timeout=60)
        r.raise_for_status()
        return r.json()

    def create_job(self, video_id: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"video_id": video_id, "extra": extra or {}}
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
        run_id: Optional[str] = None,
        dedupe: bool = True,
    ) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "top_k": int(top_k),
            "video_id": video_id,
            "run_id": run_id,
            "dedupe": bool(dedupe),
        }
        r = requests.post(self._url("/v1/search"), json=payload, timeout=120)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and "hits" in js and isinstance(js["hits"], list):
            return js["hits"]
        return []
