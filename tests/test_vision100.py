import unittest

from scripts.evaluate_vision100 import assess, camera_result, new_layouts, polygon_box


class VisionEvidenceTests(unittest.TestCase):
    def test_second_prediction_of_same_object_cannot_increment_recall(self):
        box=[10,10,12,12]
        r=assess([box,box],[box])
        self.assertEqual((r["tp"],r["fp"],r["fn"],r["unique_correct"]),(1,1,0,False))

    def test_single_prediction_without_any_target_is_a_false_unique(self):
        r=assess([[10,10,2,2]],[])
        self.assertEqual((r["tp"],r["fp"],r["fn"],r["unique_wrong"]),(0,1,0,True))

    def test_raster_polygon_endpoints_are_inclusive(self):
        self.assertEqual(polygon_box([[1,5],[2,5],[1,7]]),[1,5,2,3])

    def test_unique_station_requires_unique_object_even_if_one_slot_is_known(self):
        obj={"slot":"A","xy_mm":[215,77]}
        r=camera_result([obj,{"slot":None}],None,[215,77],False)
        self.assertTrue(r["withheld"])
        self.assertNotIn("xy_error_mm",r)
        self.assertTrue(camera_result([obj],None,[215,77],False)["false_slot"])

    def test_seed_fixes_one_hundred_distinct_scene_variations(self):
        layouts=list(new_layouts())
        self.assertEqual(layouts,list(new_layouts()))
        self.assertEqual(len({c[0] for c in layouts}),100)
        self.assertEqual(len({(c[2].xy,c[2].yaw) for c in layouts}),100)
