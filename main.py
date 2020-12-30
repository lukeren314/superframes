from super_resolution import SuperResolution
from video_converter import VideoConverter
from batch_loader import BatchLoader
INPUT_FILE = "./input1.mp4"
TEMP_DIR = "./temp"
FRAMES_DIR = "./frames"
OUTPUT_FILE = "./output.mp4"
CHECKPOINT_DIR = "./model/TecoGAN"
BATCH_SIZE = 100
BATCH_DIR = "./batch"

if __name__ == "__main__":
    vc = VideoConverter()
    # vc.video_to_frames(INPUT_FILE, TEMP_DIR, shrink_factor=2)

    # bl = BatchLoader(TEMP_DIR, BATCH_DIR, BATCH_SIZE, trim_excess=True)
    # bl.next_batch()
    # sr = SuperResolution(CHECKPOINT_DIR, BATCH_DIR)
    # sr.enhance(BATCH_DIR, FRAMES_DIR)

    # while not bl.finished:
    #     sr.enhance(BATCH_DIR, FRAMES_DIR)
    #     bl.next_batch()
    # sr.enhance(TEMP_DIR, FRAMES_DIR)
    vc.frames_to_video(FRAMES_DIR, OUTPUT_FILE)
    # vc.frames_to_video("./input", "./output1.mp4")
    print("Done!")
