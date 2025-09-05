#!/usr/bin/env python3
"""Generate pseudo-labels using YOLOv8 and save COCO-style annotations.
Reads defaults from `configs/config.yaml` via `src.config.load_config()`.
"""
import os
import json
import cv2
from ultralytics import YOLO
import supervision as sv
from src.config import load_config


def pseudo_label(video_path, out_frames, out_ann, sample_rate=2, model_name="yolov8n.pt"):
    os.makedirs(out_frames, exist_ok=True)
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)
    annotated_dir = os.path.join(out_frames, 'annotated')
    os.makedirs(annotated_dir, exist_ok=True)

    # carrega YOLOv8
    model = YOLO(model_name)

    # abre vídeo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / float(sample_rate))))
    frame_idx, saved_idx = 0, 0

    annotations = {"images": [], "annotations": [], "categories": []}
    categories = {}
    ann_id = 1

    # --- mini-SORT like tracker (lightweight, no external deps) ---
    class Track:
        def __init__(self, bbox, track_id):
            # bbox: (x1,y1,x2,y2)
            self.bbox = bbox
            x1, y1, x2, y2 = bbox
            self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.velocity = (0.0, 0.0)
            self.track_id = track_id
            self.age = 0
            self.time_since_update = 0

        def predict(self):
            # simple constant velocity prediction
            cx, cy = self.centroid
            vx, vy = self.velocity
            pred_cx = cx + vx
            pred_cy = cy + vy
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            self.centroid = (pred_cx, pred_cy)
            # rebuild bbox centered at predicted centroid
            self.bbox = (pred_cx - w/2.0, pred_cy - h/2.0, pred_cx + w/2.0, pred_cy + h/2.0)
            self.age += 1
            self.time_since_update += 1

        def update(self, bbox):
            # update using simple velocity smoothing
            x1, y1, x2, y2 = bbox
            new_cx = (x1 + x2) / 2.0
            new_cy = (y1 + y2) / 2.0
            old_cx, old_cy = self.centroid
            # velocity update (alpha smoothing)
            alpha = 0.6
            vx = alpha * (new_cx - old_cx) + (1 - alpha) * self.velocity[0]
            vy = alpha * (new_cy - old_cy) + (1 - alpha) * self.velocity[1]
            self.velocity = (vx, vy)
            self.centroid = (new_cx, new_cy)
            w = x2 - x1
            h = y2 - y1
            self.bbox = (new_cx - w/2.0, new_cy - h/2.0, new_cx + w/2.0, new_cy + h/2.0)
            self.time_since_update = 0

    def iou(bb_test, bb_gt):
        # bb: x1,y1,x2,y2
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        inter = w * h
        area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    tracks = []  # list of Track
    next_track_id = 1
    max_age = 8  # frames to keep alive without updates
    iou_threshold = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            img_name = f"frame_{saved_idx:05d}.jpg"
            img_path = os.path.join(out_frames, img_name)

            # inferência YOLO
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Prepare drawing on a copy
            vis = frame.copy()

            # build list of current detections
            dets = []
            for box, cls, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
                x1, y1, x2, y2 = map(float, box)
                dets.append({
                    'box': (x1, y1, x2, y2),
                    'cat_id': int(cls),
                    'score': float(conf)
                })

            # Predict existing tracks
            for tr in tracks:
                tr.predict()

            # Matching: compute IoU matrix and greedily match highest IoU first
            matches = []  # list of (track_idx, det_idx)
            if len(tracks) > 0 and len(dets) > 0:
                iou_mat = [[0.0 for _ in dets] for _ in tracks]
                for ti, tr in enumerate(tracks):
                    for di, d in enumerate(dets):
                        iou_mat[ti][di] = iou(tr.bbox, d['box'])

                # greedy matching
                used_t = set()
                used_d = set()
                # flatten indices by iou desc
                pairs = []
                for ti in range(len(tracks)):
                    for di in range(len(dets)):
                        pairs.append((iou_mat[ti][di], ti, di))
                pairs.sort(key=lambda x: x[0], reverse=True)
                for score, ti, di in pairs:
                    if score < iou_threshold:
                        break
                    if ti in used_t or di in used_d:
                        continue
                    used_t.add(ti)
                    used_d.add(di)
                    matches.append((ti, di))

            matched_det_indices = set(di for _, di in matches)

            # Update matched tracks
            for ti, di in matches:
                tr = tracks[ti]
                d = dets[di]
                tr.update(d['box'])

            # Create new tracks for unmatched detections
            for di, d in enumerate(dets):
                if di in matched_det_indices:
                    continue
                tr = Track(d['box'], next_track_id)
                next_track_id += 1
                tracks.append(tr)

            # Remove stale tracks
            new_tracks = []
            for tr in tracks:
                if tr.time_since_update <= max_age:
                    new_tracks.append(tr)
            tracks = new_tracks

            # draw detections and write annotations from current tracks
            ann_before = len(annotations["annotations"])
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                w, h = x2 - x1, y2 - y1
                # find the original detection matching this track (by IoU) to get class/score if available
                best_det = None
                best_iou = 0.0
                for d in dets:
                    cur_iou = iou(tr.bbox, d['box'])
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_det = d

                if best_det is None:
                    # skip annotation writing for tracks without recent detection
                    continue

                cat_id = best_det['cat_id']
                conf = best_det['score']

                if cat_id not in categories:
                    categories[cat_id] = results.names[int(cat_id)]

                annotations["annotations"].append({
                    "id": ann_id,
                    "image_id": saved_idx,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf),
                    "iscrowd": 0,
                    "track_id": int(tr.track_id)
                })
                ann_id += 1

                # draw box
                color = (0, 255, 0)
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
                label = f"{categories[cat_id]}:{tr.track_id} {conf:.2f}"
                cv2.putText(vis, label, (x1i, max(15, y1i-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # only save and register this image if new annotations were added
            ann_after = len(annotations["annotations"])
            if ann_after > ann_before:
                annotated_path = os.path.join(annotated_dir, img_name)
                cv2.imwrite(annotated_path, vis)
                # also save original frame if desired
                cv2.imwrite(img_path, frame)

                annotations["images"].append({
                    "id": saved_idx,
                    "file_name": img_name,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0])
                })

                saved_idx += 1
        frame_idx += 1

    # adiciona categorias
    annotations["categories"] = [{"id": cid, "name": name} for cid, name in categories.items()]

    with open(out_ann, "w", encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    cap.release()
    print(f"Pseudo-labeling concluído. {saved_idx} frames processados.")
    print(f"Anotações salvas em: {out_ann}")
    return out_ann


if __name__ == '__main__':
    cfg = load_config()
    video = cfg.get('video_path')
    frames_dir = cfg.get('frames_dir')
    ann_path = os.path.join(frames_dir, 'annotations.json')
    if not video or not frames_dir:
        print('Configuração incompleta. Defina video_path e frames_dir em configs/config.yaml')
    else:
        pseudo_label(video, frames_dir, ann_path, sample_rate=cfg.get('sample_rate', 2))
