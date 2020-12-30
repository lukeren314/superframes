from lib.super_resolution import SuperResolution
from lib.video_converter import VideoConverter
from lib.batch_loader import BatchLoader

INPUT_FILE = "./input1.mp4"
TEMP_DIR = "./temp"
FRAMES_DIR = "./frames"
OUTPUT_FILE = "./output.mp4"
CHECKPOINT_DIR = "./model/TecoGAN"
BATCH_SIZE = 100
BATCH_DIR = "./batch"

if __name__ == "__main__":
    vc = VideoConverter()
    vc.frames_to_video(FRAMES_DIR, OUTPUT_FILE)
    print("Done!")
