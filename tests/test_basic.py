"""
基础测试
"""
import unittest
from pathlib import Path

class TestCogAtom(unittest.TestCase):
    
    def test_import(self):
        """测试包导入"""
        try:
            from cogatom import CogAtomPipeline
            self.assertTrue(True)
        except ImportError:
            self.fail("无法导入CogAtomPipeline")
    
    def test_data_structure(self):
        """测试数据目录结构"""
        data_dir = Path("data")
        self.assertTrue(data_dir.exists())
        self.assertTrue((data_dir / "samples").exists())
        self.assertTrue((data_dir / "samples" / "sample_seed.jsonl").exists())

if __name__ == "__main__":
    unittest.main()
