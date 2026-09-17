import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark


class BenchmarkProcessSamplingTests(unittest.TestCase):
    def test_get_tree_keeps_root_when_child_enumeration_is_sandboxed(self) -> None:
        root = Mock()
        root.is_running.return_value = True
        root.children.side_effect = PermissionError(
            1,
            "Operation not permitted (originated from sysctl())",
        )

        with patch("benchmark.psutil.Process", return_value=root):
            processes = benchmark.get_tree(12345)

        self.assertEqual(processes, [root])

    def test_get_tree_ignores_processes_that_fail_running_check(self) -> None:
        root = Mock()
        root.is_running.return_value = True
        inaccessible_child = Mock()
        inaccessible_child.is_running.side_effect = OSError(
            1,
            "Operation not permitted",
        )
        root.children.return_value = [inaccessible_child]

        with patch("benchmark.psutil.Process", return_value=root):
            processes = benchmark.get_tree(12345)

        self.assertEqual(processes, [root])

    def test_process_sampling_skips_bare_os_errors(self) -> None:
        readable_process = Mock(pid=101)
        readable_process.cpu_times.return_value = SimpleNamespace(
            user=1.25,
            system=0.75,
        )
        readable_process.memory_info.return_value = SimpleNamespace(rss=4096)
        readable_process.name.return_value = "ocr-worker"

        inaccessible_process = Mock(pid=102)
        inaccessible_process.cpu_times.side_effect = OSError(
            1,
            "Operation not permitted",
        )

        with patch(
            "benchmark.get_tree",
            return_value=[readable_process, inaccessible_process],
        ):
            cpu_time, rss, names = benchmark.get_cpu_time_and_memory(12345)

        self.assertEqual(cpu_time, 2.0)
        self.assertEqual(rss, 4096)
        self.assertEqual(names, ["101:ocr-worker"])


if __name__ == "__main__":
    unittest.main()
