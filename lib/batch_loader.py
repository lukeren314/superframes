import os
import shutil

class BatchLoader:
    def __init__(self, file_dir, batch_dir, batch_size, trim_excess=False):
        self.file_dir = file_dir
        self.batch_dir = batch_dir
        self.batch_size = batch_size
        self.batch_index = 0
        self.file_list = sorted([img for img in os.listdir(file_dir)])
        self.batch_num = len(self.file_list) // self.batch_size
        self.finished = False
    def next_batch(self):
        if os.path.exists(self.batch_dir):
            for img in os.listdir(self.batch_dir):
                os.remove(os.path.join(self.batch_dir, img))
        else:
            os.mkdir(self.batch_dir)
        if self.batch_index == self.batch_num:
            finished = True
            return
        for i in range(self.batch_index*self.batch_size, (self.batch_index+1)*self.batch_size):
            if i == len(self.file_list):
                break
            shutil.copyfile(os.path.join(self.file_dir, self.file_list[i]), os.path.join(self.batch_dir, self.file_list[i]))
        self.batch_index += 1