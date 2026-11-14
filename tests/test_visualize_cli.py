"""验证可视化入口的导入路径，不读取或下载数据集。"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualizeCliTests(unittest.TestCase):
    """覆盖模块运行和编辑器常用的直接运行文件方式。"""

    def test_help_from_module_and_script(self) -> None:
        """入口不依赖额外 PYTHONPATH，直接运行也不依赖当前工作目录。"""
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as other_directory:
            cases = (
                (PROJECT_ROOT, ["-m", "src.visualize"]),
                (PROJECT_ROOT, ["src/visualize.py"]),
                (other_directory, [str(PROJECT_ROOT / "src" / "visualize.py")]),
            )
            for cwd, arguments in cases:
                with self.subTest(cwd=str(cwd), arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, *arguments, "--help"],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("--seed", result.stdout)


if __name__ == "__main__":
    unittest.main()
