import openai
import os

openai.api_type = "azure"
openai.api_base = os.getenv("CONNECTION")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("KEY")
import import_tweets
import import_racist_data
import import_ethos_data


import re


rating = 7
sample_size = 10
shuffle = False
threshold = 0.5


def main():
  pos, neg = dataset_provider("ethos")

  print("negatives:")
  run_statistics(neg)

  print("positive:")
  run_statistics(pos)

def create_prompt(s):
  return "you are an offensive text classifire rating text beween 1 and 10,  return 1 if text rate above {} and 0 otherwise, classify the following text: \"{}\"".format(rating, re.escape(s))


def run_prompt(s):
  prompt=create_prompt(s)
  return int(openai.Completion.create(
  engine="Test",
  prompt=prompt,
  temperature=0.7,
  max_tokens=400,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=None)['choices'][0]['text'].lstrip())



def run_openai(inputs):
  error_texts = []
  wrong_texts = []

  for index, row in inputs.iterrows():
    try:
      text = row['input']
      lable_output = row['class']
      ai_output = run_prompt(text)
      if ai_output != lable_output:
        wrong_texts.append(text)
    except Exception as a:
      # print(a)
      error_texts.append(text)


  return wrong_texts, error_texts

def run_statistics(inputs):
  wrong_texts, error_texts = run_openai(inputs)
  size=inputs.size
  wrong_size = len(wrong_texts)
  error_size = len(error_texts)

  print("success: {}/{}".format(size - wrong_size - error_size, size))
  print("success (without errors): {}/{}".format(size - wrong_size - error_size, size - error_size))

  print("wrong: {}/{}".format(wrong_size, size))
  print("error: {}/{}".format(error_size, size))
  print("error texts")
  print(error_texts)


def dataset_provider(name):
  match name:
    case 'tweets':
      return import_tweets.get_inputs(limit=sample_size, shuffle=shuffle)

    case 'racist':
       return import_racist_data.get_inputs(limit=sample_size, shuffle=shuffle)

    case 'ethos':
      return import_ethos_data.get_inputs(limit=sample_size, shuffle=shuffle, threshold=threshold)

  return;

main()


