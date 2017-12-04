from window_functions import *

def detect_vehicle(img):

    ystart = 400
    ystop = 656
    scale = 1.5

    bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    out_img = draw_bboxes(img, bboxes)

    # Add heat to each box in box list
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, bboxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


if __name__ == '__main__':

    from moviepy.editor import VideoFileClip

    output_video = 'car_video_output.mp4'
    # clip1 = VideoFileClip("challenge_video.mp4")

    clip1 = VideoFileClip("project_video.mp4")
    clip = clip1.fl_image(detect_vehicle)
    clip.write_videofile(output_video, audio=False)
