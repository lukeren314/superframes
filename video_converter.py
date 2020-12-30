import cv2
import os

class VideoConverter:
  def __init__(self):
    pass
  def frames_to_video(self, image_dir, output_file, FPS=30):
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape
    print("Combining frames, total %d" % len(images))
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"MP4V"), FPS, (width,height))

    for image in images:
      video.write(cv2.imread(os.path.join(image_dir, image)))
  
    cv2.destroyAllWindows()
    video.release()

  def video_to_frames(self, input_file, output_dir, shrink_factor = 1):
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    for img in os.listdir(output_dir):
      os.remove(os.path.join(output_dir, img))
    vidcap = cv2.VideoCapture(input_file)
    success, image = vidcap.read()
    count = 0
    while success:
      print("Converting frame %d" % count)
      height, width, layers = image.shape
      if shrink_factor != 1:
        image = cv2.resize(image, (width // shrink_factor, height // shrink_factor))
      cv2.imwrite(os.path.join(output_dir,f"frame{count:05d}.png"), image)
      success, image = vidcap.read()
      count += 1
