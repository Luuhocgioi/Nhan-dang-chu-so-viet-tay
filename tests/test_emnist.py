import unittest
import numpy as np
import cv2
from src.emnist_model import normalize_glyph,CHARACTERS
from src.emnist_predict import prepare_page
from src.config import ROOT


class EMNISTPreprocessingTests(unittest.TestCase):
    def test_character_mapping(self):
        self.assertEqual(CHARACTERS[:10],'0123456789')
        self.assertEqual(len(set(CHARACTERS)),62)

    def test_normalization_preserves_tall_shape(self):
        a=np.zeros((120,200),np.uint8);a[10:110,90:100]=255
        b=normalize_glyph(a);ys,xs=np.where(b>0)
        self.assertEqual(b.shape,(28,28))
        self.assertEqual(ys.max()-ys.min()+1,20)
        self.assertLess(xs.max()-xs.min(),4)

    def test_faint_purple_retained_wood_excluded(self):
        page=np.full((200,300,3),200,np.uint8)
        page[:,:50]=(80,120,170) # nền nâu BGR
        cv2.line(page,(140,40),(140,160),(175,140,175),6)
        _,mask,mode=prepare_page(page)
        self.assertEqual(mode,'color')
        self.assertEqual(int(mask[:,:45].sum()),0)
        self.assertGreater(mask[:,130:150].sum(),0)

    def test_tile_borders_excluded_from_foreground(self):
        page=np.zeros((160,500,3),np.uint8)
        for x in (20,180,340):
            cv2.rectangle(page,(x,20),(x+110,130),(210,210,210),-1)
            cv2.line(page,(x+55,45),(x+55,105),(10,10,10),7)
        _,mask,mode=prepare_page(page)
        self.assertEqual(mode,'tiles')
        self.assertEqual(int(mask[20:25,:].sum()),0)
        self.assertGreater(mask[50:100,70:80].sum(),0)

    def test_too_few_bright_cells_do_not_trigger_tiles(self):
        from src.emnist_predict import tiled_foreground
        image=np.zeros((160,500),np.uint8)
        cv2.rectangle(image,(20,20),(130,130),210,-1)
        self.assertIsNone(tiled_foreground(image))

    def test_red_ink_keeps_dark_outline(self):
        page=np.full((220,300,3),220,np.uint8)
        cv2.line(page,(130,60),(130,140),(20,20,20),9)
        cv2.line(page,(130,60),(130,140),(20,20,180),3)
        _,mask,mode=prepare_page(page)
        self.assertEqual(mode,'color')
        self.assertGreater(mask[70:130,126:135].sum(),0)
        # Vùng nét đen ở ngoài lõi đỏ phải còn, nền giấy không thành nét.
        self.assertGreater(mask[100,127],0)
        self.assertEqual(int(mask[190:210,10:50].sum()),0)

    def test_dense_page_fragment_merge_preserves_pixels(self):
        from src.emnist_predict import merge_small_fragments
        mask=np.zeros((280,400),np.uint8)
        for y in range(10,250,30):
            for x in range(10,380,30):cv2.rectangle(mask,(x,y),(x+7,y+17),255,-1)
        mask[40:58,40:48]=0
        mask[40:46,40:48]=255
        mask[48:58,40:48]=255
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        merged,new_stats=merge_small_fragments(labels,stats)
        self.assertEqual(merged[42,42],merged[53,42])
        self.assertNotEqual(merged[42,42],merged[73,42])
        np.testing.assert_array_equal(labels>0,merged>0)


@unittest.skipUnless((ROOT/'models/emnist_candidate.json').exists(),'Train EMNIST first')
class EMNISTInferenceTests(unittest.TestCase):
    def test_blank_returns_empty_result(self):
        from src.emnist_predict import EMNISTPredictor
        from src.preprocess import EmptyImageError
        model=EMNISTPredictor()
        try:
            result,_,thumbs=model.recognize(np.full((200,300,3),255,np.uint8))
        except EmptyImageError:
            return
        self.assertEqual(result['text'],'')
        self.assertEqual(thumbs,[])
