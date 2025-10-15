import cv2
import numpy as np
import math


# 반경 기준 (픽셀)
D_THRESH = 100
CLASS_COLORS  = {
    "tree":   (255,   0,   0),
    "branch": (  0, 255,   0),
    "root":   (  0,   0, 255),
    "crown":  (255,   0, 255),
    "fruit":  (  0, 255, 255),
    "gnarl":  (128,   0, 128),
}

# 전처리 ================================
def preprocess_contour(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(gray)
    cv2.drawContours(canvas, cnts, -1, 255, 1)
    thick  = cv2.dilate(canvas, np.ones((2,2), np.uint8), iterations=1)
    return cv2.cvtColor(cv2.bitwise_not(thick), cv2.COLOR_GRAY2BGR)

def preprocess_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th   = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    thick = cv2.dilate(th, np.ones((2,2), np.uint8), iterations=1)
    return cv2.cvtColor(cv2.bitwise_not(thick), cv2.COLOR_GRAY2BGR)

# 후처리 =================================

# 두 박스 간 최소 거리 계산
def box_to_box_dist(b1, b2):
    """
    두 박스 b1, b2 의 테두리 사이 최소 거리
    b = (x1,y1,x2,y2)
    """
    x11,y11,x12,y12 = b1
    x21,y21,x22,y22 = b2
    # 수평으로 겹치면 dx = 0, 아니면 gap
    dx = max(x21 - x12, x11 - x22, 0)
    dy = max(y21 - y12, y11 - y22, 0)
    return math.hypot(dx, dy)

def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

# tree: confidence가 가장 높은 1개만 유지
# branch / gnarl / fruit / root / crown: 모두 confidence가 가장 높은 1개만 유지
# crown 기본 모드는 "best" (즉, 위/아래 위치 무시하고 conf 최고 1개)
def filter_predictions(
    boxes: np.ndarray,
    confs: np.ndarray,
    clses: np.ndarray,
    names: list,
    crown_mode: str = "best"   # 기본을 "best" 로 변경
) -> list:
    """
    :param boxes: (N,4) array of [x1,y1,x2,y2]
    :param confs: (N,) confidence scores
    :param clses: (N,) class indices
    :param names: model.names 리스트
    :param crown_mode: "best" or "above" (기본 "best")
    :return: sorted list of indices to keep
    """
    keep = set()

    # 1) tree: 최고 conf 1개
    tree_idxs = [i for i, c in enumerate(clses) if names[c] == "tree"]
    if tree_idxs:
        best_tree = max(tree_idxs, key=lambda i: confs[i])
        keep.add(best_tree)

    # 2) 각 클래스별로 최고 conf 1개만 유지
    #    (branch, gnarl, fruit, root, crown)
    for cls_name in ("branch", "gnarl", "fruit", "root", "crown"):
        idxs = [i for i, c in enumerate(clses) if names[c] == cls_name]
        if not idxs:
            continue

        if cls_name == "crown" and crown_mode == "above" and tree_idxs:
            # 선택: tree 위쪽에 있는 crown만 고려
            # tree 중심 y 계산
            bt = best_tree
            y_center_tree = (boxes[bt, 1] + boxes[bt, 3]) / 2
            # tree 위쪽(cy < ty)만 필터
            above_idxs = [
                i for i in idxs
                if ((boxes[i, 1] + boxes[i, 3]) / 2) < y_center_tree
            ]
            idxs = above_idxs or idxs  # 위쪽이 없으면 전체에서 선택

        # 이 시점의 idxs 중에서 conf 최고 1개
        best_i = max(idxs, key=lambda i: confs[i])
        keep.add(best_i)

    # 3) 나머지 클래스 (만약 더 있으면) 모두 최고 1개만
    #    (위에서 다뤘다면 pass)

    return sorted(keep)