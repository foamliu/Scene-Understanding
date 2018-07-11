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
kernel = 3
weight_decay = 1e-2

folder_metadata = 'data/SUNRGBDtoolbox/Metadata/'
folder_2D_segmentation = 'annotation2Dfinal'
folder_rgb_image = 'image'
seg_path = 'data/SUNRGBD2Dseg/'

seg37list = hdf5storage.loadmat('seg37list.mat')['seg37list'][0]
seg37list = [seg[0] for seg in seg37list]
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

objectColors = ['#000000', '#2523ad', '#066ff1', '#abebbb', '#64071f', '#e6840a', '#ccf808', '#636371', '#9d0eea',
                '#007567', '#a20084', '#4c7f57', '#e20724', '#0f5406', '#fd4465', '#75ce74', '#25cb7e', '#f62d45',
                '#fcda9f', '#f7a337', '#44e9c0', '#0abc21', '#330d75', '#0f18bf', '#5e7f46', '#f417c0', '#b57670',
                '#dcae65', '#79dd07', '#ce6f38', '#7143a4', '#61632d', '#80ea5b', '#27a355', '#09cb29', '#5f989e',
                '#c08035', '#6a1446']
colors = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in objectColors]

crop_size = 512
