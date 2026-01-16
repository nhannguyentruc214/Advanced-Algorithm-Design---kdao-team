import anthropic
import time
import pandas as pd
import requests
import pickle
import os
import numpy as np

API_KEY = '<API_KEY>'
client = anthropic.Anthropic(api_key=API_KEY)

file_path = "../data/netflix/"
max_threads = 5
cnt = 0

def construct_prompting(item_attribute, item_list):
    history_string = "User history:\n"
    for index in item_list:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(year) + ", "
        history_string += title + "\n"
    output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt += history_string
    prompt += output_format
    return prompt

def file_reading():
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict_claude','rb'))
    return augmented_attribute_dict

def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt):
    try:
        augmented_user_profiling_dict = file_reading()
    except pickle.UnpicklingError as e:
        print("Error occurred while unpickling:", e)
        return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)

    if index in augmented_user_profiling_dict:
        return 0
    else:
        try:
            print(f"{index}")
            prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index])

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )
            content = message.content[0].text

            print(f"content: {content}, model_type: {model_type}")

            augmented_user_profiling_dict[index] = content
            pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict_claude','wb'))
            error_cnt = 0

        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        except EOFError as ke:
            print("EOFError: : Ran out of input error occurred while accessing the response:", str(ke))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
        return 1


error_cnt = 0

toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year', 'title'])
augmented_user_profiling_dict = {}
if os.path.exists(file_path + "augmented_user_profiling_dict_claude"):
    print(f"The file augmented_user_profiling_dict_claude exists.")
    augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict_claude','rb'))
else:
    print(f"The file augmented_user_profiling_dict_claude does not exist.")
    pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict_claude','wb'))

adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y

for index in range(0, len(adjacency_list_dict.keys())):
    print(index)
    re = LLM_request(toy_item_attribute, adjacency_list_dict, index, "claude-haiku", augmented_user_profiling_dict, error_cnt)


augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict_claude','rb'))
augmented_user_init_embedding = {}
if os.path.exists(file_path + "augmented_user_init_embedding_claude"):
    print(f"The file augmented_user_init_embedding_claude exists.")
    augmented_user_init_embedding = pickle.load(open(file_path + 'augmented_user_init_embedding_claude','rb'))
else:
    print(f"The file augmented_user_init_embedding_claude does not exist.")
    pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding_claude','wb'))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

for index in augmented_user_profiling_dict.keys():
    if index in augmented_user_init_embedding:
        continue

    profile_text = str(augmented_user_profiling_dict[index])
    embedding = model.encode(profile_text)
    augmented_user_init_embedding[index] = np.array(embedding)
    print(index)

    pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding_claude','wb'))


augmented_user_init_embedding = pickle.load(open(file_path + 'augmented_user_init_embedding_claude','rb'))
augmented_user_init_embedding_list = []
for i in range(len(augmented_user_init_embedding)):
    augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
pickle.dump(augmented_user_init_embedding_final, open(file_path + 'augmented_user_init_embedding_final_claude','wb'))
