from config import num_classes, seg38list
from data_generator import get_image, get_semantic

if __name__ == '__main__':
    counts = dict()
    for i in range(len(num_classes)):
        counts[i] = 0

    filename = '{}_names.txt'.format('train')
    with open(filename, 'r') as f:
        names = f.read().splitlines()
    total = 0
    for name in names:
        image, image_size = get_image(name)
        semantic = get_semantic(name, image_size)
        h, w = semantic.shape[:2]

        for r in range(h):
            for c in range(w):
                counts[semantic[r, c]] += 1
        total += h * w

    for i in range(len(num_classes)):
        class_name = seg38list[i]
        num_pixels = counts[i]
        percent = num_pixels / total
        print('class_name: {}, num_pixels: {}, perentage: {:.4%}'.format(class_name, num_pixels, percent * 100))
