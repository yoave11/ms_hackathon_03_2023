#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://connectorgpt.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "c7870848f2f64fa79b48358c9d660c8b"
import import_data
import re


rating = 7

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

sample_size=10

print("negatives:")
run_statistics(import_data.neg_input(limit=sample_size, shuffle=False))

print("positive:")
run_statistics(import_data.pos_input(limit=sample_size, shuffle=False))
# print(run_prompt(s))


