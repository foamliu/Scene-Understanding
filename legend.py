import cv2 as cv
import numpy as np

from config import seg37list, colors, num_classes
from utils import draw_str

if __name__ == '__main__':
    num_rows, num_cols = 5, 8
    height_cell, width_cell = 50, 100
    margin_x, margin_y = 5, 15

    frame = np.zeros((num_rows * height_cell, num_cols * width_cell, 3), np.uint8)

    for i in range(num_rows):
        for j in range(num_cols):
            id = i * num_cols + j
            print(id)
            if id <= num_classes - 1:
                color = colors[id]
                if id == num_classes - 1:
                    name = 'other'
                else:
                    name = seg37list[id][0]
                print(name)
                top = i * height_cell
                left = j * width_cell
                bottom = top + (height_cell - 1)
                right = left + (width_cell - 1)
                cv.rectangle(frame, (left, top), (right, bottom), color, cv.FILLED)
                draw_str(frame, (left + margin_x, top + margin_y), name)

    cv.imshow('frame', frame)
    cv.imwrite('images/legend.png', frame)
    cv.waitKey(0)
