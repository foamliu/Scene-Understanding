import hdf5storage

img_rows, img_cols = 320, 320
channel = 3
batch_size = 40
epochs = 1000
patience = 50
num_samples = 10335
num_train_samples = 8268
# num_samples - num_train_samples
num_valid_samples = 2067
num_classes = 38

folder_metadata = 'data/SUNRGBDtoolbox/Metadata/'
folder_2D_segmentation = 'annotation2Dfinal'
folder_rgb_image = 'image'

num_samples = 10335
seg_path = 'data/SUNRGBD2Dseg'

seg37list = hdf5storage.loadmat('seg37list.mat')['seg37list'][0]
# print(seg37list)
# ['wall',
#  'floor',
#  'cabinet',
#  'bed',
#  'chair',
#  'sofa',
#  'table',
#  'door',
#  'window',
#  'bookshelf',
#  'picture',
#  'counter',
#  'blinds',
#  'desk',
#  'shelves',
#  'curtain',
#  'dresser',
#  'pillow',
#  'mirror',
#  'floor_mat',
#  'clothes',
#  'ceiling',
#  'books',
#  'fridge',
#  'tv',
#  'paper',
#  'towel',
#  'shower_curtain',
#  'box',
#  'whiteboard',
#  'person',
#  'night_stand',
#  'toilet',
#  'sink',
#  'lamp',
#  'bathtub',
#  'bag']


objectColors = ['#000000', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',
                '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                '#17becf', '#9edae5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69',
                '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#621e15', '#e59076', '#128dcd', '#083c52',
                '#64c5f2', '#61afaf', '#0f7369', '#9c9da1', '#365e96', '#983334', '#77973d', '#5d437c', '#36869f',
                '#d1702f', '#8197c5', '#c47f80', '#acc484', '#9887b0', '#2d588a', '#58954c', '#e9a044', '#c12f32',
                '#723e77', '#7d807f', '#9c9ede', '#7375b5', '#4a5584', '#cedb9c', '#b5cf6b', '#8ca252', '#637939',
                '#e7cb94', '#e7ba52', '#bd9e39', '#8c6d31', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#de9ed6',
                '#ce6dbd', '#a55194', '#7b4173', '#000000', '#0000FF']
colors = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in objectColors]
