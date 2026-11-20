import json
import os
import re
import tempfile
import unittest
import zipfile
from io import BytesIO

os.environ.setdefault("MUSIC_STUDIO_OUT", tempfile.mkdtemp(prefix="music-studio-test-"))

try:
    from kaggle import music_worker as worker
except ImportError:  # requests is not installed
    worker = None
from scripts.push_music_worker import inject_job


REQUEST_MARKER = "<!--music-request-->"


def parse_request_body(body: str):
    index = body.find(REQUEST_MARKER)
    if index < 0:
        return None
    raw = body[index + len(REQUEST_MARKER):].strip()
    raw = raw.split("<!--/music-request-->")[0].strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def parse_music_meta(body: str):
    match = re.search(r"<!--music-meta:(.*?)-->", body, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except ValueError:
        return None


class ProtocolTests(unittest.TestCase):
    def test_worker_job_injection_preserves_future_import_position(self):
        worker_source = (
            "# -*- coding: utf-8 -*-\n"
            '"""Private worker."""\n'
            "from __future__ import annotations\n"
            "VALUE = 1\n"
        )
        assembled = inject_job(worker_source, {"gh_repo": "owner/repo"})
        compile(assembled, "music_worker.py", "exec")
        self.assertLess(
            assembled.index("from __future__ import annotations"),
            assembled.index("JOB = "),
        )

    def test_parse_request_created_by_browser(self):
        payload = {
            "version": 1,
            "prompt": "Hebrew pop with piano and strings",
            "lyrics": "[Verse 1]\n...",
            "language": "he",
            "duration": 180,
        }
        body = (
            "Music request\n\n"
            + REQUEST_MARKER
            + "\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n<!--/music-request-->"
        )
        self.assertEqual(parse_request_body(body), payload)

    def test_reject_missing_or_malformed_request(self):
        self.assertIsNone(parse_request_body("plain issue"))
        self.assertIsNone(parse_request_body(REQUEST_MARKER + "\n{broken}"))

    def test_gemini_planner_falls_back_when_secret_is_not_configured(self):
        if worker is None:
            self.skipTest("requests is not installed")
        original = worker.GEMINI_API_KEY
        worker.GEMINI_API_KEY = ""
        try:
            self.assertEqual(worker.gemini_plan("instrumental piano", "", 120, "en"), {})
        finally:
            worker.GEMINI_API_KEY = original

    def test_parse_release_metadata(self):
        meta = {
            "version": 2,
            "prompt": "More piano, calmer drums",
            "lyrics": "text",
            "zip_url": "https://github.com/example/release.zip",
        }
        body = "<!--music-meta:" + json.dumps(meta, ensure_ascii=False) + "-->"
        self.assertEqual(parse_music_meta(body), meta)

    def test_bundle_contains_player_audio_and_metadata(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("song.mp3", b"fake-mp3")
            archive.writestr("prompt.txt", "prompt")
            archive.writestr("lyrics.txt", "lyrics")
            archive.writestr("metadata.json", "{}")
        with zipfile.ZipFile(BytesIO(buffer.getvalue())) as archive:
            self.assertIn("song.mp3", archive.namelist())
            self.assertIn("prompt.txt", archive.namelist())
            self.assertIn("lyrics.txt", archive.namelist())
            self.assertIn("metadata.json", archive.namelist())


def make_issue(number=7, owner="owner"):
    payload = {"version": 1, "prompt": "pop", "lyrics": "", "language": "he", "duration": 120}
    body = "req\n<!--music-request-->\n" + json.dumps(payload) + "\n<!--/music-request-->"
    return {"number": number, "body": body, "user": {"login": owner}, "updated_at": "t1", "comments": 0}


def assistant_comment(version=1, feedback_id=None):
    meta = "<!--music-meta:" + json.dumps({"version": version, "prompt": "p", "lyrics": "l"}) + "-->"
    marker = f"<!--music-feedback-result:{feedback_id}-->" if feedback_id else worker.STARTED_MARKER
    return {"id": 100 + version, "user": {"login": "github-actions[bot]"},
            "body": f"{worker.ASSISTANT_MARKER}\nready\n{meta}\n{marker}"}


def feedback_comment(comment_id, text, author="owner"):
    return {"id": comment_id, "user": {"login": author}, "body": worker.FEEDBACK_MARKER + "\n" + text}


@unittest.skipIf(worker is None, "requests is not installed")
class WorkerTaskTests(unittest.TestCase):
    def setUp(self):
        worker.GH_OWNER = "owner"
        worker.SEEN.clear()

    def test_new_request_is_a_task(self):
        task = worker.find_task(make_issue(), [])
        self.assertIsNotNone(task)
        self.assertEqual(task["feedback"], "")
        self.assertIsNone(task["previous"])

    def test_finished_request_without_feedback_is_idle(self):
        self.assertIsNone(worker.find_task(make_issue(), [assistant_comment()]))

    def test_owner_feedback_becomes_task_with_previous_meta(self):
        comments = [assistant_comment(), feedback_comment(200, "more piano")]
        task = worker.find_task(make_issue(), comments)
        self.assertEqual(task["feedback"], "more piano")
        self.assertEqual(task["feedback_id"], 200)
        self.assertEqual(task["previous"]["version"], 1)

    def test_feedback_from_other_users_is_ignored(self):
        comments = [assistant_comment(), feedback_comment(201, "hack", author="stranger")]
        self.assertIsNone(worker.find_task(make_issue(), comments))

    def test_processed_feedback_is_not_repeated(self):
        comments = [assistant_comment(), feedback_comment(202, "x"), assistant_comment(2, 202)]
        self.assertIsNone(worker.find_task(make_issue(), comments))

    def test_next_task_skips_unchanged_issues(self):
        issue = make_issue()
        calls = []
        original = worker.issue_comments
        worker.issue_comments = lambda n: calls.append(n) or [assistant_comment()]
        try:
            self.assertIsNone(worker.next_task([issue]))
            self.assertIsNone(worker.next_task([issue]))
        finally:
            worker.issue_comments = original
        self.assertEqual(calls, [7])


if __name__ == "__main__":
    unittest.main()
