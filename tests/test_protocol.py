import json
import re
import unittest
import zipfile
from io import BytesIO


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


if __name__ == "__main__":
    unittest.main()
