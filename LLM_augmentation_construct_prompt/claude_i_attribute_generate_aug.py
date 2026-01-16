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

def construct_prompting(item_attribute, indices):
    pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
    item_list_string = ""
    for index in indices:
        year = item_attribute.loc[index, 'year']
        title = item_attribute.loc[index, 'title']
        item_list_string += "["
        item_list_string += str(index)
        item_list_string += "] "
        item_list_string += str(year) + ", "
        item_list_string += title + "\n"
    output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    prompt = pre_string + item_list_string + output_format
    return prompt


def LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt):
    if indices[0] in augmented_attribute_dict:
        return 0
    else:
        try:
            print(f"{indices}")
            prompt = construct_prompting(toy_item_attribute, indices)

            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )
            content = message.content[0].text

            print(f"content: {content}, model_type: {model_type}")

            rows = content.strip().split("\n")
            for i,row in enumerate(rows):
                elements = row.split("::")
                director = elements[0].strip()
                country = elements[1].strip()
                language = elements[2].strip()
                augmented_attribute_dict[indices[i]] = {}
                augmented_attribute_dict[indices[i]][0] = director
                augmented_attribute_dict[indices[i]][1] = country
                augmented_attribute_dict[indices[i]][2] = language
            pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict_claude','wb'))

            error_cnt = 0
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            error_cnt += 1
            if error_cnt==5:
                return 1
            return LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            time.sleep(5)
            error_cnt += 1
            if error_cnt==5:
                return 1
            return LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            time.sleep(5)
            error_cnt += 1
            if error_cnt==5:
                return 1
            return LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            time.sleep(5)
            return 1
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            error_cnt += 1
            if error_cnt==5:
                return 1
            return LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
        return 1


error_cnt = 0

augmented_attribute_dict = {}
if os.path.exists(file_path + "augmented_attribute_dict_claude"):
    print(f"The file augmented_attribute_dict_claude exists.")
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict_claude','rb'))
else:
    print(f"The file augmented_attribute_dict_claude does not exist.")
    pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict_claude','wb'))

toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', header=None, names=['id','year', 'title'])
toy_item_attribute = toy_item_attribute.set_index('id')

for i in range(0, len(toy_item_attribute), 1):
    indices = [i]
    print(f"###i###: {i}")
    re = LLM_request(toy_item_attribute, indices, "claude-haiku", augmented_attribute_dict, error_cnt)


raw_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year','title'])
augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict_claude','rb'))
director_list, country_list, language_list = [], [], []

for i in range(len(raw_item_attribute)):
    if i in augmented_attribute_dict:
        director_list.append(augmented_attribute_dict[i][0])
        country_list.append(augmented_attribute_dict[i][1])
        language_list.append(augmented_attribute_dict[i][2])
    else:
        print(f"Item {i} missing, using defaults")
        director_list.append("Unknown")
        country_list.append("Unknown")
        language_list.append("English")

director_series = pd.Series(director_list)
country_series = pd.Series(country_list)
language_series = pd.Series(language_list)
raw_item_attribute['director'] = director_series
raw_item_attribute['country'] = country_series
raw_item_attribute['language'] = language_series
raw_item_attribute.to_csv(file_path + 'augmented_item_attribute_agg_claude.csv', index=False, header=None)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

year_embedding_dict, title_embedding_dict, director_embedding_dict, country_embedding_dict, language_embedding_dict = {}, {}, {}, {}, {}
augmented_atttribute_embedding_dict = {
    'year': year_embedding_dict,
    'title': title_embedding_dict,
    'director': director_embedding_dict,
    'country': country_embedding_dict,
    'language': language_embedding_dict
}

file_name = "augmented_atttribute_embedding_dict_claude"
if os.path.exists(file_path + file_name):
    print(f"Loading existing {file_name}...")
    augmented_atttribute_embedding_dict = pickle.load(open(file_path + file_name,'rb'))
    existing_count = sum(len(v) for v in augmented_atttribute_embedding_dict.values())
    print(f"Loaded existing embeddings: {existing_count} total")
else:
    print(f"Creating new {file_name}...")
    pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))

toy_augmented_item_attribute = pd.read_csv(file_path + 'augmented_item_attribute_agg_claude.csv', names=['id', 'year','title', 'director', 'country', 'language'])

for i in range(len(toy_augmented_item_attribute)):
    needs_processing = False
    for value in augmented_atttribute_embedding_dict.keys():
        if i not in augmented_atttribute_embedding_dict[value]:
            needs_processing = True
            break

    if not needs_processing:
        continue

    if i % 1000 == 0:
        print(f"Progress: {i}/{len(toy_augmented_item_attribute)}")

    for value in augmented_atttribute_embedding_dict.keys():
        if i in augmented_atttribute_embedding_dict[value]:
            continue

        text = str(toy_augmented_item_attribute[value][i])
        embedding = model.encode(text)
        augmented_atttribute_embedding_dict[value][i] = embedding.tolist()

    if i % 100 == 0:
        pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))

pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))

augmented_total_embed_dict = {'year':[] , 'title':[], 'director':[], 'country':[], 'language':[]}
augmented_atttribute_embedding_dict = pickle.load(open(file_path + file_name,'rb'))

for value in augmented_atttribute_embedding_dict.keys():
    for i in range(len(augmented_atttribute_embedding_dict[value])):
        augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])
    augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])
    print(f"  {value}: shape {augmented_total_embed_dict[value].shape}")

pickle.dump(augmented_total_embed_dict, open(file_path + 'augmented_total_embed_dict_claude','wb'))
