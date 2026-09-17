"""Isolated tests for Chandra's completed-result signed URL handling."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

import requests

from run_chandra import (
    RESULT_DOWNLOAD_TIMEOUT,
    bbox_of,
    html_to_text,
    poll_result,
    resolve_completed_result,
)

CHECK_URL = "https://api.datalab.to/api/v1/convert/request-123"
SIGNED_RESULT_URL = "https://storage.example.test/results/request-123?signature=private"


def json_response(payload: object) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


class ChandraResultUrlTests(unittest.TestCase):
    def test_downloads_signed_result_only_after_a_complete_poll(self) -> None:
        session = Mock()
        session.get.side_effect = [
            json_response({"status": "processing"}),
            json_response(
                {
                    "status": "complete",
                    "success": True,
                    "request_id": "request-123",
                    "result_url": SIGNED_RESULT_URL,
                }
            ),
            json_response({"markdown": "# Hindi result", "json": {"children": []}}),
        ]

        with patch("run_chandra.time.sleep") as sleep:
            result, _elapsed, polls = poll_result(
                session,
                "api-secret",
                CHECK_URL,
                poll_interval=0.1,
                timeout_seconds=30,
            )

        self.assertEqual(polls, 2)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "request-123")
        self.assertEqual(result["markdown"], "# Hindi result")
        self.assertEqual(
            session.get.call_args_list,
            [
                call(CHECK_URL, headers={"X-API-Key": "api-secret"}, timeout=(30, 120)),
                call(CHECK_URL, headers={"X-API-Key": "api-secret"}, timeout=(30, 120)),
                call(
                    SIGNED_RESULT_URL,
                    headers={},
                    timeout=RESULT_DOWNLOAD_TIMEOUT,
                    allow_redirects=False,
                ),
            ],
        )
        sleep.assert_called_once_with(0.1)

    def test_keeps_inline_completed_result_without_downloading_result_url(self) -> None:
        session = Mock()
        inline = {
            "status": "complete",
            "success": True,
            "markdown": "",
            "result_url": SIGNED_RESULT_URL,
        }

        resolved = resolve_completed_result(session, inline)

        self.assertIs(resolved, inline)
        session.get.assert_not_called()

    def test_does_not_follow_result_url_for_non_complete_terminal_status(self) -> None:
        session = Mock()
        session.get.return_value = json_response(
            {
                "status": "failed",
                "success": False,
                "result_url": SIGNED_RESULT_URL,
            }
        )

        result, _elapsed, polls = poll_result(
            session,
            "api-secret",
            CHECK_URL,
            poll_interval=0.1,
            timeout_seconds=30,
        )

        self.assertEqual(polls, 1)
        self.assertEqual(result["status"], "failed")
        session.get.assert_called_once_with(
            CHECK_URL,
            headers={"X-API-Key": "api-secret"},
            timeout=(30, 120),
        )

    def test_rejects_malformed_signed_result_url_without_requesting_it(self) -> None:
        session = Mock()

        with self.assertRaisesRegex(RuntimeError, "invalid result_url"):
            resolve_completed_result(
                session,
                {"status": "complete", "result_url": "http://example.test/result"},
            )

        session.get.assert_not_called()

    def test_explains_non_json_signed_result_download(self) -> None:
        session = Mock()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("unexpected HTML")
        session.get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
            resolve_completed_result(
                session,
                {"status": "complete", "result_url": SIGNED_RESULT_URL},
            )

    def test_explains_signed_result_download_failure(self) -> None:
        session = Mock()
        session.get.side_effect = requests.Timeout("signed URL expired")

        with self.assertRaisesRegex(RuntimeError, "may have expired"):
            resolve_completed_result(
                session,
                {"status": "complete", "result_url": SIGNED_RESULT_URL},
            )

    def test_refuses_signed_result_redirect(self) -> None:
        session = Mock()
        response = json_response({"markdown": "should not be read"})
        response.status_code = 302
        session.get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Refused a redirect"):
            resolve_completed_result(
                session,
                {"status": "complete", "result_url": SIGNED_RESULT_URL},
            )

    def test_html_text_keeps_chandras_unicode_whitespace_policy(self) -> None:
        self.assertEqual(html_to_text("<p>A&nbsp; B</p>"), "A B")

    def test_bbox_falls_back_after_an_invalid_earlier_alias(self) -> None:
        self.assertEqual(bbox_of({"bbox": [], "box": [1, 2, 3, 4]}), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
