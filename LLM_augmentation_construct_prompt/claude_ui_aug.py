import anthropic
import pickle
import numpy as np
import os
from tqdm import tqdm
import time
import pandas as pd

file_path = "../data/netflix/"
API_KEY = '<API_KEY>'

if not API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

client = anthropic.Anthropic(api_key=API_KEY)

def construct_prompting(item_attribute, item_list, candidate_list):
    history_string = "User history:\n"
    for index in item_list:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(year) + ", "
        history_string += title + "\n"
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        year = item_attribute['year'][index.item()]
        title = item_attribute['title'][index.item()]
        candidate_string += "["
        candidate_string += str(index.item())
        candidate_string += "] "
        candidate_string += str(year) + ", "
        candidate_string += title + "\n"
    output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    prompt = ""
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt

candidate_indices = pickle.load(open(file_path + 'candidate_indices','rb'))
candidate_indices_dict = {}
for index in range(candidate_indices.shape[0]):
    candidate_indices_dict[index] = candidate_indices[index]

adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y

toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id', 'year', 'title'])

augmented_sample_dict = {}
if os.path.exists(file_path + "augmented_sample_dict_claude"):
    print(f"The file augmented_sample_dict_claude exists.")
    augmented_sample_dict = pickle.load(open(file_path + 'augmented_sample_dict_claude','rb'))
else:
    print(f"The file augmented_sample_dict_claude does not exist.")
    pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict_claude','wb'))

def file_reading():
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_sample_dict_claude','rb'))
    return augmented_attribute_dict

def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict):

    try:
        augmented_sample_dict = file_reading()
    except pickle.UnpicklingError as e:
        print("Error occurred while unpickling:", e)
        LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)

    if index in augmented_sample_dict:
        return 0
    else:
        try:
            print(f"{index}")
            prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )

            content = message.content[0].text
            print(f"content: {content}, model_type: {model_type}")

            samples = content.split("::")
            pos_sample = int(samples[0])
            neg_sample = int(samples[1])
            augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample
            pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict_claude','wb'))

        except anthropic.RateLimitError as e:
            print("Rate limit error:", str(e))
            time.sleep(10)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            time.sleep(10)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            time.sleep(10)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(10)

        return 1

for index in range(0, len(adjacency_list_dict)):
    re = LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "claude-haiku", augmented_sample_dict)
    if re == 1:
        time.sleep(0.5)