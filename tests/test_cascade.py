import numpy as np

from s2m.cascade import cascade, cluster_boxes, iou
from s2m.perception import Detections


def test_overlapping_boxes_share_a_cluster():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9.5], [50, 50, 60, 60]], float)
    cid, reps = cluster_boxes(boxes, np.array([0.9, 0.5, 0.4]))
    assert cid[0] == cid[1] != cid[2] and len(reps) == 2
    assert np.isclose(iou(boxes[:1], boxes[:1])[0, 0], 1.0)


def test_verification_keeps_only_classes_clip_confirms():
    box = np.array([[0, 0, 10, 10]], np.float32)
    a = {0: Detections(box, np.array([5], np.int32), np.array([0.3], np.float32))}
    b = {0: Detections(box, np.array([7], np.int32), np.array([0.2], np.float32))}
    prompt_ids = np.array([5, 7, -1])  # two object classes and one structural prompt
    verified = {"probs": np.array([[0.7, 0.1, 0.2]], np.float16), "cluster_0": np.array([0]), "cluster_1": np.array([0])}
    out = cascade([a, b], verified, prompt_ids, tau=0.2)
    assert out[0].class_ids.tolist() == [5]
    out = cascade([a, b], verified, prompt_ids, tau=0.0)
    assert sorted(out[0].class_ids.tolist()) == [5, 7]
