import cv2
import numpy as np
import pandas as pd
import time


def calculate_jaccard_index(rect1, rect2):
    """
    Calcule l'indice de Jaccard entre deux rectangles.
    """
    left = max(rect1.bb_left, rect2.bb_left)
    right = min(rect1.bb_left + rect1.bb_width, rect2.bb_left + rect2.bb_width)
    dx = max(0, right - left)
    top = max(rect1.bb_top, rect2.bb_top)
    bottom = min(rect1.bb_top + rect1.bb_height, rect2.bb_top + rect2.bb_height)
    dy = max(0, bottom - top)
    intersection_area = dx * dy
    union_area = rect1.bb_width * rect1.bb_height + rect2.bb_width * rect2.bb_height - intersection_area
    return intersection_area / union_area


def compute_similarity_matrix(rectangles_frame1, rectangles_frame2):
    """
    Calcule la matrice de similarité entre les rectangles de deux frames successives.
    """
    num_rectangles1 = rectangles_frame1.shape[0]
    num_rectangles2 = rectangles_frame2.shape[0]
    similarity_matrix = np.zeros((num_rectangles1, num_rectangles2))

    for i in range(num_rectangles1):
        for j in range(num_rectangles2):
            similarity_matrix[i, j] = calculate_jaccard_index(rectangles_frame1.iloc[i], rectangles_frame2.iloc[j])

    return similarity_matrix


def match_tracks(similarity_matrix):
    """
    Détermine l'organisation des pistes à partir de la matrice de similarité.
    """
    track_assignments = [-1] * similarity_matrix.shape[1]
    while np.max(similarity_matrix) >= 0.1:
        max_index = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
        track_assignments[max_index[1]] = max_index[0]
        similarity_matrix[max_index[0], :] = 0
        similarity_matrix[:, max_index[1]] = 0

    return track_assignments


def get_centroid(box):
    """
    Calcule les coordonnées du centre d'un rectangle.
    """
    centroid_x = int(box.bb_left + box.bb_width / 2)
    centroid_y = int(box.bb_top + box.bb_height / 2)
    return centroid_x, centroid_y


def update_tracks_and_history(rectangles, previous_positions, previous_tracks, current_tracks, track_histories):
    """
    Met à jour les pistes et l'historique des pistes en cours.
    """
    updated_positions = previous_positions.copy()
    for i in range(len(previous_tracks)):
        if i in current_tracks:
            updated_positions[previous_positions.index(i)] = current_tracks.index(i)
            track_histories[previous_positions.index(i)].append(rectangles.iloc[current_tracks.index(i)])
        else:
            updated_positions[previous_positions.index(i)] = -1

    for i in range(len(current_tracks)):
        if current_tracks[i] == -1:
            updated_positions.append(i)
            track_histories.append([rectangles.iloc[i]])

    return updated_positions, track_histories


def process_frame(frame_number, track_positions, previous_tracks, track_histories, detections):
    """
    Traite une frame : lit l'image, crée la matrice de similarité, met à jour les pistes, et affiche les résultats.
    """
    image_path = f"ADL-Rundle-6/img1/{str(frame_number).zfill(6)}.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rectangles_previous_frame = detections[detections.frame == (frame_number - 1)]
    rectangles_current_frame = detections[detections.frame == frame_number]

    similarity_matrix = compute_similarity_matrix(rectangles_previous_frame, rectangles_current_frame)
    current_tracks = match_tracks(similarity_matrix)

    track_positions, track_histories = update_tracks_and_history(rectangles_current_frame, track_positions,
                                                                 previous_tracks, current_tracks, track_histories)

    for i in range(rectangles_current_frame.shape[0]):
        track_id = track_positions.index(i)
        box = track_histories[track_id][-1]
        cv2.rectangle(image, (int(box.bb_left), int(box.bb_top)),
                      (int(box.bb_left + box.bb_width), int(box.bb_top + box.bb_height)), (255, 0, 0), 2)
        cv2.putText(image, str(track_id), (int(box.bb_left), int(box.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    2)

        for j in range(len(track_histories[track_id]) - 1):
            start_point = get_centroid(track_histories[track_id][j])
            end_point = get_centroid(track_histories[track_id][j + 1])
            cv2.line(image, start_point, end_point, (255, 255, 0), 2)

    previous_tracks = current_tracks
    cv2.imshow('Tracks', image)
    cv2.waitKey(1)
    return track_positions, previous_tracks, track_histories


def main():
    detections = pd.read_csv('ADL-Rundle-6/det/det.txt',
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = detections.frame.min(), detections.frame.max()

    track_positions, previous_tracks, track_histories = [], [], []
    initial_rectangles = detections[detections.frame == min_frame]

    for i in range(initial_rectangles.shape[0]):
        track_positions.append(i)
        track_histories.append([initial_rectangles.iloc[i]])

    previous_tracks = track_positions.copy()
    start_time = time.time()

    for frame_number in range(min_frame + 1, max_frame + 1):
        track_positions, previous_tracks, track_histories = process_frame(frame_number, track_positions,
                                                                          previous_tracks, track_histories, detections)

    end_time = time.time()
    print(f'Secondes par frame: {(end_time - start_time) / (max_frame - min_frame):.2f}')


if __name__ == "__main__":
    main()