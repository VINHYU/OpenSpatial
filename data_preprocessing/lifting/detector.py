"""Gemini-based open-vocabulary object detection."""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from .config import LiftingConfig


class GeminiObjectDetector:
    """Detect objects with the public Gemini generateContent REST API."""

    def __init__(self, config: LiftingConfig):
        self.config = config
        self.max_workers = config.detection_workers
        self.cache_dir = os.path.join(config.output_base_dir, "detection_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, image_path: str) -> str:
        path_hash = hashlib.md5(os.path.abspath(image_path).encode()).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.cache_dir, f"{stem}_{path_hash}.json")

    def _load_from_cache(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        path = self._get_cache_path(image_path)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file).get("detections", [])
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Cache] Could not read {path}: {exc}")
            return None

    def _save_to_cache(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        img_width: int,
        img_height: int,
    ) -> None:
        path = self._get_cache_path(image_path)
        payload = {
            "image_path": image_path,
            "img_width": img_width,
            "img_height": img_height,
            "detections": detections,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _get_image_mime_type(filename: str) -> str:
        suffix = os.path.splitext(filename)[1].lower()
        return {
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

    @staticmethod
    def _extract_list_from_json(data) -> List[Dict]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("detections", "objects", "items", "results", "boxes", "predictions"):
                if isinstance(data.get(key), list):
                    return data[key]
            for value in data.values():
                if isinstance(value, list):
                    return value
        return []

    def _parse_json_response(self, raw_content: str) -> List[Dict]:
        content = raw_content.strip()
        for match in re.findall(r"```(?:json)?\s*([\s\S]*?)```", content):
            try:
                return self._extract_list_from_json(json.loads(match.strip()))
            except json.JSONDecodeError:
                pass

        for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
            for match in re.findall(pattern, content):
                try:
                    result = self._extract_list_from_json(json.loads(match))
                    if result:
                        return result
                except json.JSONDecodeError:
                    pass

        try:
            return self._extract_list_from_json(json.loads(content))
        except json.JSONDecodeError:
            print(f"[Gemini] Could not decode response: {content[:200]}...")
            return []

    def _call_api(
        self, image_path: str, img_width: int, img_height: int
    ) -> List[Dict[str, Any]]:
        model = quote(self.config.detector_model, safe="")
        endpoint = f"{self.config.api_base_url.rstrip('/')}/models/{model}:generateContent"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": self._get_image_mime_type(image_path),
                            "data": self._encode_image(image_path),
                        }
                    },
                    {"text": self.config.detection_prompt},
                ],
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }

        last_exception = None
        for attempt in range(self.config.api_max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.config.api_timeout,
                )
                if response.status_code == 429:
                    wait_time = 60 + random.uniform(0, 10)
                    print(f"[Gemini] Rate limited; retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                response.raise_for_status()
                response_json = response.json()
                raw_content = response_json["candidates"][0]["content"]["parts"][0]["text"]
                raw_detections = self._parse_json_response(raw_content)

                detections = []
                for item in raw_detections:
                    box = item.get("box_2d")
                    if not isinstance(box, list) or len(box) != 4 or "label" not in item:
                        continue
                    y0 = int(box[0] / 1000 * img_height)
                    x0 = int(box[1] / 1000 * img_width)
                    y1 = int(box[2] / 1000 * img_height)
                    x1 = int(box[3] / 1000 * img_width)
                    if y0 >= y1 or x0 >= x1:
                        continue
                    detections.append({
                        "box_2d": [x0, y0, x1, y1],
                        "label": item["label"],
                        "description": item.get("description", ""),
                    })
                return detections
            except Exception as exc:
                last_exception = exc
                if attempt + 1 < self.config.api_max_retries:
                    wait_time = min(2 ** attempt, 60) + random.uniform(0, 1)
                    print(
                        f"[Gemini] Request failed ({attempt + 1}/"
                        f"{self.config.api_max_retries}): {exc}; "
                        f"retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Gemini request failed after {self.config.api_max_retries} attempts"
        ) from last_exception

    def _detect_single(
        self, task: Tuple[int, str, int, int], use_cache: bool
    ) -> Tuple[int, str, List[Dict[str, Any]]]:
        index, image_path, width, height = task
        if use_cache:
            cached = self._load_from_cache(image_path)
            if cached is not None:
                return index, image_path, cached
        detections = self._call_api(image_path, width, height)
        self._save_to_cache(image_path, detections, width, height)
        return index, image_path, detections

    def detect_batch(
        self,
        tasks: List[Tuple[int, str, int, int]],
        use_cache: bool = False,
    ) -> Dict[int, Tuple[str, List[Dict[str, Any]]]]:
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._detect_single, task, use_cache): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    index, image_path, detections = future.result()
                except Exception as exc:
                    index, image_path = task[:2]
                    detections = []
                    print(f"[Gemini] Failed to process {image_path}: {exc}")
                results[index] = (image_path, detections)
                print(
                    f"[Gemini] {os.path.basename(image_path)}: "
                    f"{len(detections)} objects"
                )
        return results
