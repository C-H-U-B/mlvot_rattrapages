from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import pandas as pd
from KalmanFilter import KalmanFilter
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


def compute_similarity(rect1, rect2):
    """
    Calcule l'indice de Jaccard et le renvoie s'il dépasse une valeur seuil minimale.
    """
    jaccard_index = calculate_jaccard_index(rect1, rect2)
    return jaccard_index if jaccard_index >= 0.1 else 0


def create_similarity_matrix(rectangles_frame1, rectangles_frame2):
    """
    Calcule la matrice de similarité entre les rectangles de deux frames successives.
    """
    num_rectangles1 = rectangles_frame1.shape[0]
    num_rectangles2 = rectangles_frame2.shape[0]
    similarity_matrix = np.zeros((num_rectangles1, num_rectangles2))

    for i in range(num_rectangles1):
        for j in range(num_rectangles2):
            similarity_matrix[i, j] = compute_similarity(rectangles_frame1.iloc[i], rectangles_frame2.iloc[j])

    return similarity_matrix


def match_tracks(similarity_matrix):
    """
    Détermine l'organisation des pistes à partir de la matrice de similarité en utilisant la fonction linear_sum_assignment.
    """
    track_assignments = [-1] * similarity_matrix.shape[1]
    rows, cols = linear_sum_assignment(similarity_matrix, maximize=True)

    for i in range(len(rows)):
        if similarity_matrix[rows[i], cols[i]] >= 0.1:
            track_assignments[cols[i]] = rows[i]

    return track_assignments


def calculate_centroid(rect):
    """
    Calcule les coordonnées du centre d'un rectangle.
    """
    centroid_x = int(rect.bb_left + rect.bb_width / 2)
    centroid_y = int(rect.bb_top + rect.bb_height / 2)
    return centroid_x, centroid_y


def predict_next_position(rect, kalman_filters, track_id):
    """
    Prédit la prochaine position d'un rectangle en utilisant le filtre de Kalman.
    """
    col_names = list(rect.index)
    current_centroid = calculate_centroid(rect)
    kalman_update = kalman_filters[track_id].update([[current_centroid[0]], [current_centroid[1]]])
    prediction = kalman_filters[track_id].predict()
    predicted_rect = [
        rect.frame, rect.id,
        rect.bb_left + prediction[0][0] - current_centroid[0],
        rect.bb_top + prediction[1][0] - current_centroid[1],
        rect.bb_width, rect.bb_height, 1, -1, -1, -1
    ]
    predicted_df = pd.DataFrame(predicted_rect, index=col_names)
    return predicted_df[0]


def update_tracks_and_history(rectangles, previous_positions, previous_tracks, current_tracks, track_histories,
                              kalman_filters):
    """
    Met à jour les positions et l'historique des pistes en cours.
    """
    updated_positions = previous_positions.copy()
    for i in range(len(previous_tracks)):
        track_id = previous_positions.index(i)
        if i in current_tracks:
            updated_positions[track_id] = current_tracks.index(i)
            track_histories[track_id].append(
                predict_next_position(rectangles.iloc[updated_positions[track_id]], kalman_filters, track_id))
        else:
            updated_positions[track_id] = -1

    for i in range(len(current_tracks)):
        if current_tracks[i] == -1:
            updated_positions.append(i)
            current_centroid = calculate_centroid(rectangles.iloc[i])
            kalman_filters.append(KalmanFilter(
                dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1,
                base_x=current_centroid[0], base_y=current_centroid[1]
            ))
            predict_next_position(rectangles.iloc[i], kalman_filters, len(kalman_filters) - 1)
            track_histories.append([rectangles.iloc[i]])

    return updated_positions, track_histories, kalman_filters


def get_previous_positions(track_positions, track_histories):
    """
    Récupère les positions précédentes des rectangles à partir de l'historique des pistes.
    """
    previous_boxes = []
    for i in range(max(track_positions)):
        track_id = track_positions.index(i)
        previous_boxes.append(track_histories[track_id][-1])
    return pd.DataFrame(previous_boxes)


def process_frame(frame_number, track_positions, previous_tracks, track_histories, detections, kalman_filters):
    """
    Traite une frame : lit l'image, crée la matrice de similarité, met à jour les pistes, et affiche les résultats.
    """
    image_path = f"ADL-Rundle-6/img1/{str(frame_number).zfill(6)}.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    previous_rectangles = get_previous_positions(track_positions, track_histories)
    current_rectangles = detections[detections.frame == frame_number]

    similarity_matrix = create_similarity_matrix(previous_rectangles, current_rectangles)
    current_tracks = match_tracks(similarity_matrix)

    track_positions, track_histories, kalman_filters = update_tracks_and_history(
        current_rectangles, track_positions, previous_tracks, current_tracks, track_histories, kalman_filters
    )

    # Afficher les prédictions pour les pistes précédentes
    for i in range(previous_rectangles.shape[0]):
        rect = previous_rectangles.iloc[i]
        centroid = calculate_centroid(rect)
        cv2.rectangle(image, (int(rect.bb_left), int(rect.bb_top)),
                      (int(rect.bb_left + rect.bb_width), int(rect.bb_top + rect.bb_height)), (0, 255, 0), 2)
        cv2.circle(image, centroid, 5, (0, 255, 0), -1)

    # Afficher les rectangles actuels et l'historique des pistes
    for i in range(current_rectangles.shape[0]):
        track_id = track_positions.index(i)
        rect = track_histories[track_id][-1]
        cv2.rectangle(image, (int(rect.bb_left), int(rect.bb_top)),
                      (int(rect.bb_left + rect.bb_width), int(rect.bb_top + rect.bb_height)), (255, 0, 0), 2)
        cv2.putText(image, str(track_id), (int(rect.bb_left), int(rect.bb_top)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        for j in range(len(track_histories[track_id]) - 1):
            cv2.line(image, calculate_centroid(track_histories[track_id][j]),
                     calculate_centroid(track_histories[track_id][j + 1]), (255, 255, 0), 2)

    previous_tracks = current_tracks
    cv2.imshow('Trackers', image)
    cv2.waitKey(1)
    return track_positions, previous_tracks, track_histories, kalman_filters


def save_tracks_to_file(detections, track_histories):
    """
    Enregistre les pistes dans un fichier suivant le format du fichier de vérité terrain.
    """
    with open("ADL-Rundle-6_tp4.txt", 'w') as file:
        for track_id, track in enumerate(track_histories):
            for rect in track:
                file.write(
                    f"{rect.frame},{track_id},{rect.bb_left},{rect.bb_top},{rect.bb_width},{rect.bb_height},1,-1,-1,-1\n")


def main():
    detections = pd.read_csv('ADL-Rundle-6/det/det.txt',
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = detections.frame.min(), detections.frame.max()

    track_positions, previous_tracks, track_histories, kalman_filters = [], [], [], []
    initial_rectangles = detections[detections.frame == min_frame]

    # Initialiser les premières pistes
    for i in range(initial_rectangles.shape[0]):
        track_positions.append(i)
        initial_centroid = calculate_centroid(initial_rectangles.iloc[i])
        kalman_filters.append(KalmanFilter(
            dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1,
            base_x=initial_centroid[0], base_y=initial_centroid[1]
        ))
        predict_next_position(initial_rectangles.iloc[i], kalman_filters, len(kalman_filters) - 1)
        track_histories.append([initial_rectangles.iloc[i]])

    previous_tracks = track_positions.copy()
    start_time = time.time()

    # Traiter chaque frame
    for frame_number in range(min_frame + 1, max_frame + 1):
        track_positions, previous_tracks, track_histories, kalman_filters = process_frame(
            frame_number, track_positions, previous_tracks, track_histories, detections, kalman_filters
        )

    end_time = time.time()
    print('Secondes par frame: ', (end_time - start_time) / (max_frame - min_frame))

    # Enregistrer les résultats
    save_tracks_to_file(detections, track_histories)


if __name__ == "__main__":
    main()
