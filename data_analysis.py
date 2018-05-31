from config import num_classes, seg38list
from data_generator import get_image, get_semantic
from console_progressbar import ProgressBar
import numpy as np

if __name__ == '__main__':
    counts = dict()
    for class_id in range(num_classes):
        counts[class_id] = 0

    filename = '{}_names.txt'.format('train')
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    pb = ProgressBar(total=len(names), prefix='Analyzing train images', suffix='', decimals=3, length=50, fill='=')

    total = 0
    for i in range(len(names)):
        name = names[i]
        image, image_size = get_image(name)
        semantic = get_semantic(name, image_size)
        h, w = image_size
        total += h * w

        for class_id in range(num_classes):
            mat = (semantic == class_id).astype(np.float32)
            counts[class_id] += np.sum(mat)

        pb.print_progress_bar(i + 1)

    print('total: {}'.format(total))
    for class_id in range(num_classes):
        class_name = seg38list[class_id]
        num_pixels = counts[class_id]
        percent = num_pixels / total
        print('class_name: {}, num_pixels: {}, percent: {:.4%}'.format(class_name, num_pixels, percent * 100))
