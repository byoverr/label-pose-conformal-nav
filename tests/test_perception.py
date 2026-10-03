import numpy as np

from s2m.perception import Detections, load_masks, pixel_labels, save_masks, upsample_masks


def test_mask_cache_roundtrip_is_exact_on_the_stride_grid(tmp_path):
    rng = np.random.default_rng(0)
    masks = {0: rng.random((3, 48, 64)) > 0.5, 8: np.zeros((0, 48, 64), bool)}
    save_masks(tmp_path / "m.npz", masks, stride=4)
    back, stride = load_masks(tmp_path / "m.npz")
    assert stride == 4 and back[8].shape == (0, 12, 16)
    up = upsample_masks(back[0], stride, (48, 64))
    assert up.shape == (3, 48, 64)
    assert np.array_equal(up[:, ::4, ::4], masks[0][:, ::4, ::4])


def test_pixel_labels_most_confident_detection_wins_and_masks_replace_boxes():
    depth = np.full((10, 10), 2.0)
    depth[0, 0] = 0.0  # no depth: never labelled
    dets = Detections(np.array([[0, 0, 10, 10], [0, 0, 5, 5]], np.float32),
                      np.array([1, 2], np.int32), np.array([0.3, 0.9], np.float32))
    lab, score = pixel_labels(dets, depth)
    assert lab[1, 1] == 2 and score[1, 1] == np.float32(0.9)
    assert lab[8, 8] == 1
    masks = np.zeros((2, 10, 10), bool)
    masks[0, 6:, 6:] = True
    masks[1, 0:2, 0:2] = True
    lab, _ = pixel_labels(dets, depth, masks=masks)
    assert lab[8, 8] == 1 and lab[1, 1] == 2 and lab[4, 4] == -1 and lab[0, 0] == -1
