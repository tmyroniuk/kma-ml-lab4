import configparser

training_config = configparser.ConfigParser()
training_config.read('../config/training-v3.ini')

base_config = configparser.ConfigParser()
base_config.read('../config/config.ini')

# Training
test_size = float(training_config['Training']['test_size'])
use_feature_selection = training_config.getboolean('Training', 'use_feature_selection')
features_to_keep = float(training_config['Training']['features_to_keep'])
varience_threshold = float(training_config['Training']['varience_threshold'])
num_clusters = int(training_config['Training']['num_clusters'])
objective = training_config['Training']['objective']
eval_metric = training_config['Training']['eval_metric']
alpha = float(training_config['Training']['alpha'])
gamma = float(training_config['Training']['gamma'])
eta = float(training_config['Training']['eta'])
max_depth = int(training_config['Training']['max_depth'])
subsample = float(training_config['Training']['subsample'])
colsample_bytree = float(training_config['Training']['colsample_bytree'])
scale_pos_weight = int(training_config['Training']['scale_pos_weight'])
min_child_weight = int(training_config['Training']['min_child_weight'])

random_state = int(base_config['Training']['random_state'])

# Data
data_file = base_config['Data']['data_file']
test_file = base_config['Data']['test_file']
submission_file = base_config['Data']['submission_file']
features_matrix_file = base_config['Data']['features_matrix_file']
pca_result_file = base_config['Data']['pca_result_file']

# Model
tfidf_vectorizer_model = base_config['Model']['tfidf_vectorizer_model']
pca_model = base_config['Model']['pca_model']
scaler_model = base_config['Model']['scaler_model']
selector_model = base_config['Model']['selector_model']
kmeans_model = base_config['Model']['kmeans_model']
xgb_model = base_config['Model']['xgb_model']
last_model_version = base_config['Model']['last_model_version']
preprocessor_model = base_config['Model']['preprocessor_model']

xgb_model_version = int(training_config['Model']['xgb_model_version'])

