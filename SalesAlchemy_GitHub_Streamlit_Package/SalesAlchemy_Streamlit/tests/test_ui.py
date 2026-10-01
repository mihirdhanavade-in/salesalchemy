import os
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_demo_generation_and_comparison(self):
        app_path=Path(__file__).resolve().parents[1]/'app.py'
        with patch.dict(os.environ,{'HF_TOKEN':'','HF_MODEL_ID':''}):
            app=AppTest.from_file(str(app_path)).run(timeout=30)
            self.assertFalse(list(app.exception))
            next(b for b in app.button if b.label=='Generate proposal').click().run(timeout=30)
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.session_state['bundle']['sections']),5)
            next(b for b in app.button if b.label=='Run BASELINE vs ENGINEERED').click().run(timeout=30)
            self.assertFalse(list(app.exception))
            self.assertEqual(set(app.session_state['comparison']),{'Baseline','Engineered'})
            identity=app.session_state['bundle']['id']
            next(c for c in app.checkbox if c.key=='review_'+identity).check().run(timeout=30)
            self.assertEqual(app.session_state['bundle']['id'],identity)

if __name__=='__main__':
    unittest.main()
