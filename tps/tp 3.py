from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import pandas as pd
import time


def calculate_jaccard_index(rect1, rect2):
    left_intersection = max(rect1.bb_left, rect2.bb_left)
    right_intersection = min(rect1.bb_left + rect1.bb_width, rect2.bb_left + rect2.bb_width)
    dx = max(0, right_intersection - left_intersection)
    top_intersection = max(rect1.bb_top, rect2.bb_top)
    bottom_intersection = min(rect1.bb_top + rect1.bb_height, rect2.bb_top + rect2.bb_height)
    dy = max(0, bottom_intersection - top_intersection)

    intersection_area = dx * dy
    union_area = rect1.bb_width * rect1.bb_height + rect2.bb_width * rect2.bb_height - intersection_area

    return intersection_area / union_area


def compute_similarity(rect1, rect2):
    jaccard = calculate_jaccard_index(rect1, rect2)
    return jaccard if jaccard >= 0.1 else 0


def generate_similarity_matrix(d1, d2):
    similarity_matrix = np.zeros((d1.shape[0], d2.shape[0]))
    for i in range(d1.shape[0]):
        for j in range(d2.shape[0]):
            similarity_matrix[i, j] = compute_similarity(d1.iloc[i], d2.iloc[j])
    return similarity_matrix


def match_tracks(similarity_matrix):
    assignments = [-1] * len(similarity_matrix[0])
    row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)
    for i in range(len(row_indices)):
        if similarity_matrix[row_indices[i], col_indices[i]] >= 0.1:
            assignments[col_indices[i]] = row_indices[i]
    return assignments


def calculate_centroid(box):
    return int(box.bb_left + box.bb_width / 2), int(box.bb_top + box.bb_height / 2)


def update_tracks(current_detections, previous_positions, previous_tracks, new_assignments, track_histories):
    updated_positions = previous_positions.copy()
    for track_index in range(len(previous_tracks)):
        pos_index = previous_positions.index(track_index)
        if track_index in new_assignments:
            updated_positions[pos_index] = new_assignments.index(track_index)
            track_histories[pos_index].append(current_detections.iloc[updated_positions[pos_index]])
        else:
            updated_positions[pos_index] = -1
    for detection_index in range(len(new_assignments)):
        if new_assignments[detection_index] == -1:
            updated_positions.append(detection_index)
            track_histories.append([current_detections.iloc[detection_index]])
    return updated_positions, track_histories


def process_frame(frame_num, track_positions, previous_tracks, track_histories, detections):
    image = cv2.imread(f"ADL-Rundle-6/img1/{str(frame_num).zfill(6)}.jpg", cv2.IMREAD_COLOR)
    previous_detections = detections[detections.frame == (frame_num - 1)]
    current_detections = detections[detections.frame == frame_num]
    similarity_matrix = generate_similarity_matrix(previous_detections, current_detections)
    new_tracks = match_tracks(similarity_matrix)
    track_positions, track_histories = update_tracks(current_detections, track_positions, previous_tracks, new_tracks,
                                                     track_histories)

    for i in range(current_detections.shape[0]):
        track_id = track_positions.index(i)
        box = track_histories[track_id][-1]
        cv2.rectangle(image, (int(box.bb_left), int(box.bb_top)),
                      (int(box.bb_left + box.bb_width), int(box.bb_top + box.bb_height)), (255, 0, 0), 5)
        cv2.putText(image, str(track_id), (int(box.bb_left), int(box.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    1, cv2.LINE_AA)
        for j in range(len(track_histories[track_id]) - 1):
            cv2.line(image, calculate_centroid(track_histories[track_id][j]),
                     calculate_centroid(track_histories[track_id][j + 1]), (255, 255, 0), 4)

    previous_tracks = new_tracks
    cv2.imshow('tracks', image)
    cv2.waitKey(1)
    return track_positions, previous_tracks, track_histories


def save_tracking_results(detections, track_histories):
    with open("ADL-Rundle-6_tp3.txt", 'w') as file:
        min_frame, max_frame = min(detections.frame), max(detections.frame)
        min_index, max_index = 0, 0
        for frame in range(min_frame, max_frame + 1):
            while max_index < len(track_histories) and track_histories[max_index][0].frame <= frame:
                max_index += 1
            for i in range(min_index, max_index):
                track_index = frame - int(track_histories[i][0].frame)
                if track_index < len(track_histories[i]):
                    box = track_histories[i][track_index]
                    file.write(f"{frame},{i},{box.bb_left},{box.bb_top},{box.bb_width},{box.bb_height},1,-1,-1,-1\n")
                else:
                    if i == min_index:
                        min_index += 1


def main():
    detections = pd.read_csv('ADL-Rundle-6/det/det.txt',
                             names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    min_frame, max_frame = min(detections.frame), max(detections.frame)
    track_positions, previous_tracks, track_histories = [], [], []

    initial_detections = detections[detections.frame == min_frame]
    for i in range(initial_detections.shape[0]):
        track_positions.append(i)
        track_histories.append([initial_detections.iloc[i]])
    previous_tracks = track_positions.copy()

    start_time = time.time()
    for frame_num in range(min_frame + 1, max_frame + 1):
        track_positions, previous_tracks, track_histories = process_frame(frame_num, track_positions, previous_tracks,
                                                                          track_histories, detections)
    end_time = time.time()

    print('FPS:', (end_time - start_time) / (max_frame - min_frame))
    save_tracking_results(detections, track_histories)


if __name__ == "__main__":
    main()
