import pickle
import sys

from xgboost import DMatrix, Booster

sys.path.append('./src')
from config import *
from utils import Preprocessor

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def predict_toxicity(prompt, model, preprocessor):
    input_ids = preprocessor.transform(prompt)
    dtest = DMatrix(input_ids)
    output = model.predict(dtest)
    return output

if __name__ == "__main__":
    preprocessor = load_pickle(f'{preprocessor_model}')
    model = Booster()
    model.load_model(f'./{xgb_model}-v{last_model_version}.json')

    if len(sys.argv) < 2:
        print("Usage: python prediction_script.py <your_command_line_argument>")
    else:
        prompt = ' '.join(sys.argv[2:])
        print("Your prompt:", prompt)
        toxicity = predict_toxicity(prompt, model, preprocessor)
        print(f'toxic\t{toxicity[0]}\nsevere_toxic\t{toxicity[1]}\nobscene\t{toxicity[2]}\nthreat\t{toxicity[3]}\ninsult\t{toxicity[4]}\nidentity_hate\t{toxicity[5]}')