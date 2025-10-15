import cv2
import numpy as np

def compute_iou(boxA, boxB):
    # boxA, boxB 는 (x1, y1, x2, y2) 형태의 튜플
    # 교차 영역 좌표 계산
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # 교차 영역 넓이
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스 면적
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU = 교집합 / (합집합)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# 모델 예측 결과 얻기
def GetModelPredictResult(model, img_path, conf_thresh = 0.25, iou_thresh = 0.45):
    result = model.predict(
        source=img_path,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )[0]

    return result

# 원본 이미지와 모델 예측 결과를 이용해 예측 박스 그리기
def DrawBox(originalImage, modelPredictResult, classNames, color_map):
    boxes   = modelPredictResult.boxes.xyxy.cpu().numpy()
    classes = modelPredictResult.boxes.cls.cpu().numpy()
    confs   = modelPredictResult.boxes.conf.cpu().numpy()
    out     = originalImage.copy()

    for (x1,y1,x2,y2), cls, conf in zip(boxes, classes, confs):
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        label = f"{classNames[int(cls)]}:{conf:.2f}"
        color = color_map.get(classNames[int(cls)], (255,255,255))

        # -- 두껍고 부드러운 테두리 --
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # -- 폰트 설정 --
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5         # 기본 0.5 → 0.8로 키움
        thickness = 1           # 기본 1 → 2로 키움
        (tw, th), _ = cv2.getTextSize(label, font, fontScale, thickness)

        # 텍스트 배경(박스) 그리기
        ty = max(y1 - 10, th + 10)
        cv2.rectangle(
            out,
            (x1, ty - th - 4),
            (x1 + tw + 4, ty),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA
        )
        # 텍스트 그리기 (흰색)
        cv2.putText(
            out,
            label,
            (x1 + 2, ty - 2),
            font,
            fontScale,
            (255,255,255),
            thickness,
            lineType=cv2.LINE_AA
        )

    return out