from pre_processing import extract_frames_and_flow
from model_training import train_model
from post_processing import extract_and_chronologically_stack_features, perform_PCA
from repetitions_counter import count_repetitions


if __name__=='__main__':
    video_path = "./videos/demo.mp4"
    epochs_nb = 50 # generally stagnates after 50~60 epochs (for videos)
    force_reset = True # if one wants to do everything again despite part of the functions already complete

    frames_and_flow_path = extract_frames_and_flow(video_path, force_reset)
    model = train_model(frames_and_flow_path, force_reset, print_losses=True, epochs_nb=epochs_nb)
    stacked_features = extract_and_chronologically_stack_features(model, frames_and_flow_path, force_reset)
    signal_1d = perform_PCA(stacked_features)
    periods_nb = count_repetitions(signal_1d, period_range=0.10, N=4)

    print(f"There are {periods_nb} in the video \"{video_path}\".")