import torch
import torchreid
import torchvision
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import pandas as pd
from KalmanFilter import KalmanFilter
import time



# Fonction pour obtenir les détections de YOLO
def get_yolo_detections(img):
    # Charger le modèle YOLOv5
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = yolo_model(img)
    detections = results.xyxy[0].cpu().numpy()
    return detections


# Fonction pour obtenir les embeddings avec OsNet
def get_osnet_embedding(image, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    patch = image[y:y + h, x:x + w]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return None
    patch = cv2.resize(patch, (128, 256))  # Dimensions standard pour OsNet
    patch = np.expand_dims(patch, axis=0)
    patch = torch.from_numpy(patch).float().permute(0, 3, 1, 2)  # Conversion en tensor torch
    with torch.no_grad():
        embedding = model(patch.cuda()).cpu().numpy()
    return embedding


# Reste du code inchangé
def embeddings_comparisons(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0
    euclidean_distance = np.linalg.norm(emb1 - emb2) / len(emb1[0])
    return 1 - euclidean_distance if euclidean_distance < 1 else 0


def jaccard_index(d1, d2):
    left = max(d1[0], d2[0])
    right = min(d1[2], d2[2])
    dx = max(0, right - left)
    top = max(d1[1], d2[1])
    down = min(d1[3], d2[3])
    dy = max(0, down - top)
    intersect = dx * dy
    union = (d1[2] - d1[0]) * (d1[3] - d1[1]) + (d2[2] - d2[0]) * (d2[3] - d2[1]) - intersect
    return intersect / union


def elimination_by_size_ratio(d1, d2):
    area_1 = (d1[2] - d1[0]) * (d1[3] - d1[1])
    area_2 = (d2[2] - d2[0]) * (d2[3] - d2[1])
    area_1, area_2 = min(area_1, area_2), max(area_1, area_2)
    return 0 if area_2 / area_1 >= 2 else 1


def similarity_function(d1, d2, emb1, emb2):
    jaccard = jaccard_index(d1, d2)
    distance = embeddings_comparisons(emb1, emb2)
    elimination = elimination_by_size_ratio(d1, d2)
    result = jaccard * 0.9 + distance * 0.1
    return 0 if result < 0.1 or distance < 0.8 or elimination == 0 else result


def get_similarity_matrix(d1, d2, emb1, emb2):
    matrix = np.zeros((d1.shape[0], d2.shape[0]))
    for i in range(d1.shape[0]):
        for j in range(d2.shape[0]):
            matrix[i, j] = similarity_function(d1.iloc[i], d2.iloc[j], emb1[i], emb2[j])
    return matrix


def track_matching(matrix):
    new_track = [-1] * len(matrix[0])
    row, col = linear_sum_assignment(matrix)
    for i in range(len(row)):
        if matrix[row[i], col[i]] >= 0.1:
            new_track[col[i]] = row[i]
    return new_track


def get_centroid(box):
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


def get_prediction(box, kalman_filters, id):
    coord = get_centroid(box)
    kalman_filters[id].update([[coord[0]], [coord[1]]])
    prediction = kalman_filters[id].predict()
    result = [box[4], box[5], box[0] + prediction[0][0] - coord[0], box[1] + prediction[1][0] - coord[1],
              box[2] - box[0], box[3] - box[1], 1, -1, -1, -1]
    return pd.Series(result)


def track_management(d, old_pos, old_track, new_track, track_history, kalman_filters):
    track_pos = old_pos.copy()
    for i in range(len(old_track)):
        id = old_pos.index(i)
        if i in new_track:
            track_pos[id] = new_track.index(i)
            track_history[id].append(get_prediction(d.iloc[track_pos[id]], kalman_filters, id))
        else:
            track_pos[id] = -1

    for i in range(len(new_track)):
        if new_track[i] == -1:
            track_pos.append(i)
            coord = get_centroid(d.iloc[i])
            kalman_filters.append(
                KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1, base_x=coord[0],
                             base_y=coord[1]))
            get_prediction(d.iloc[i], kalman_filters, len(kalman_filters) - 1)
            track_history.append([d.iloc[i]])

    return track_pos, track_history, kalman_filters


def get_previous_positions(track_pos, track_history):
    boxes = [track_history[track_pos.index(i)][-1] for i in range(max(track_pos))]
    return pd.DataFrame(boxes)


def treatment(f, track_pos, old_track, track_history, det, kalman_filters, old_emb):
    img = cv2.imread(f"ADL-Rundle-6/img1/{str(f).zfill(6)}.jpg", cv2.IMREAD_COLOR)
    yolo_detections = get_yolo_detections(img)
    d1 = get_previous_positions(track_pos, track_history)
    d2 = pd.DataFrame(yolo_detections, columns=['bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf', 'class'])
    d2 = d2[d2['class'] == 0]  # Filtrer les piétons détectés

    embeddings = [get_osnet_embedding(img, d2.iloc[j]) for j in range(d2.shape[0])]

    matrix = get_similarity_matrix(d1, d2, old_emb, embeddings)
    track = track_matching(matrix)
    track_pos, track_history, kalman_filters = track_management(d2, track_pos, old_track, track, track_history,
                                                                kalman_filters)

    for i in range(d1.shape[0]):
        box = d1.iloc[i]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, f"{i}", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i in range(d2.shape[0]):
        box = d2.iloc[i]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(img, f"{i}", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    old_track = track
    cv2.imshow('tracks', img)
    cv2.waitKey(1)
    return track_pos, old_track, track_history, kalman_filters, embeddings


def save_track(det, track_history):
    with open("ADL-Rundle-6_tp5_osnet.txt", 'w') as file:
        min_frame, max_frame = det.frame.min(), det.frame.max()
        min_i, max_i = 0, 0

        for f in range(min_frame, max_frame + 1):
            while max_i < len(track_history) and track_history[max_i][0].frame <= f:
                max_i += 1
            for i in range(min_i, max_i):
                index = f - int(track_history[i][0].frame)
                if index < len(track_history[i]):
                    box = track_history[i][index]
                    file.write(f"{f},{i},{box[0]},{box[1]},{box[2]},{box[3]},1,-1,-1,-1\n")
                else:
                    if i == min_i:
                        min_i += 1


def main():
    # Charger les données de réidentification
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data/Market-1501-v15.09.15",
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"]
    )

    # Construire le modèle, l'optimiseur et le planificateur de taux d'apprentissage
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        loss="softmax",
        pretrained=False  # Désactiver le téléchargement automatique
    )
    # Charger le modèle préentraîné
    state_dict = torch.load("reid-data/models/osnet_x1_0.pth")
    model.load_state_dict(state_dict, strict=False)

    # Remplacer les couches de classification
    num_classes = datamanager.num_train_pids
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    # Construire le moteur pour l'entraînement et les tests
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # Exécuter l'entraînement et les tests
    engine.run(
        save_dir="log/osnet",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )
    det = pd.read_csv('ADL-Rundle-6/det/det.txt',
                      names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = det.frame.min(), det.frame.max()
    track_pos, old_track, track_history, kalman_filters = [], [], [], []
    d = det[det.frame == min_frame]

    img = cv2.imread(f"ADL-Rundle-6/img1/{str(min_frame).zfill(6)}.jpg", cv2.IMREAD_COLOR)
    embeddings = [get_osnet_embedding(img, d.iloc[i]) for i in range(d.shape[0])]

    for i in range(d.shape[0]):
        track_pos.append(i)
        coord = get_centroid(d.iloc[i])
        kalman_filters.append(
            KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1, base_x=coord[0],
                         base_y=coord[1]))
        get_prediction(d.iloc[i], kalman_filters, len(kalman_filters) - 1)
        track_history.append([d.iloc[i]])

    old_track = track_pos.copy()
    start_time = time.time()

    for f in range(min_frame + 1, max_frame + 1):
        track_pos, old_track, track_history, kalman_filters, embeddings = treatment(f, track_pos, old_track,
                                                                                    track_history, det, kalman_filters,
                                                                                    embeddings)

    end_time = time.time()
    print('Secondes par frame:', (end_time - start_time) / (max_frame - min_frame))

    save_track(det, track_history)


if __name__ == "__main__":
    main()
