import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import prepare_data

class PrepareDataTests(unittest.TestCase):
    def test_local_crop_uses_recorded_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'data').mkdir()
            source=root/'dataset/images/train2017'
            source.mkdir(parents=True)
            Image.new('RGB',(40,20),'red').save(source/'42.jpg')
            records=[{'id':'42_1','image_id':'42','bbox_xyxy':[10,5,30,15]}]
            (root/'data/metadata.json').write_text(json.dumps({'records':records}))
            with patch.object(prepare_data,'ROOT',root):
                prepare_data.write_thumbnails(root/'dataset')
            with Image.open(root/'data/thumbnails/42_1.jpg') as result:
                self.assertEqual(result.size,(20,10))
                red,green,blue=result.getpixel((5,5))
                self.assertGreater(red,240)
                self.assertLess(green,10)
                self.assertLess(blue,10)

if __name__=='__main__': unittest.main()
