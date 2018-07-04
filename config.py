import hdf5storage

img_rows, img_cols = 256, 256
channel = 3
batch_size = 16
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
seg_path = 'data/SUNRGBD2Dseg/'

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

# all =  [seg37list, other]

objectColors = ['#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#36869f',
                '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                '#17becf', '#9edae5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69',
                '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#000000']
colors = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in objectColors]
