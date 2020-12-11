# dataset version
cp_data = True
version = 'v2'
assert version in ['v1', 'v2'], 'wrong dataset version'

# training set 
train_set = 'train'
assert train_set in ['train', 'train+val'], 'wrong training set'

# running settings
entropy = 4.5
scale = 32
use_cos = True

# image feature type
image_feature = 'grid'
assert image_feature in ['rcnn', 'grid'], 'wrong image feature type'

# paths
main_path = '/disk0/vqa/test/vqa-data/'
rcnn_path = main_path + '/rcnn-data-count/'
grid_path = main_path + '/grid-data/'

qa_path = main_path + 'vqa-cp/' if cp_data else main_path
qa_path += version # questions and answers
vocabulary_path = qa_path + 'vocab.json'

bottom_up_trainval_path = main_path + 'bottom_up_feature/trainval_resnet101_faster_rcnn_genome_36.tsv'
bottom_up_test_path = main_path + 'bottom_up_feature/test2015_resnet101_faster_rcnn_genome_36.tsv'
rcnn_trainval_path = rcnn_path + 'genome-trainval.h5'
rcnn_test_path = rcnn_path + 'genome-test.h5'

image_train_path = main_path + 'mscoco/train2014'
image_val_path = main_path + 'mscoco/val2014'
image_test_path = main_path + 'mscoco/test2015'
grid_trainval_path = grid_path + 'resnet-trainval.h5'
grid_test_path = grid_path + 'resnet-test.h5'

# import settings
dataset = 'mscoco'
max_question_len = 15
task = 'OpenEnded' if not cp_data else 'vqacp'
test_split = 'test-dev2015'  # either 'test-dev2015' or 'test2015'

# text pretrained model config
pretrained_model = 'glove'
glove_path = main_path + 'word_embed/glove/'
glove_path_filtered = qa_path + 'glove_filter'
text_embed_size = 300

# preprocess image config
rcnn_output_size = 36  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal
preprocess_batch_size = 8
image_size = 448  # scale shorter end of image to this size and centre crop
grid_output_size = 14 * 14  # size of the feature maps after processing through a network
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 30
batch_size = 256
initial_lr = 1e-3
data_workers = 4
max_answers = 3000
