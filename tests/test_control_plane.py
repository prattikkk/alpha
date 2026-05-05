import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import control_plane


class ControlPlaneTests(unittest.TestCase):
    def test_state_and_command_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "runtime_control.json"
            queue_path = Path(td) / "runtime_commands.jsonl"

            with mock.patch.object(control_plane, "CONTROL_STATE_PATH", state_path), mock.patch.object(
                control_plane, "COMMAND_QUEUE_PATH", queue_path
            ):
                state = control_plane.get_control_state()
                self.assertFalse(state["paused"])

                state = control_plane.set_paused(True)
                self.assertTrue(state["paused"])

                state = control_plane.update_control_state(
                    overrides={"min_confidence": 0.8, "max_correlated_positions": 2}
                )
                self.assertEqual(state["overrides"]["min_confidence"], 0.8)
                self.assertEqual(state["overrides"]["max_correlated_positions"], 2)

                command = control_plane.enqueue_command("close_symbol", {"symbol": "BTCUSDT"})
                self.assertEqual(command["action"], "close_symbol")

                drained = control_plane.drain_commands()
                self.assertEqual(len(drained), 1)
                self.assertEqual(drained[0]["payload"]["symbol"], "BTCUSDT")

                # Queue should be empty after drain.
                self.assertEqual(control_plane.drain_commands(), [])


if __name__ == "__main__":
    unittest.main()
