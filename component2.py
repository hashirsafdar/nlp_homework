import numpy as np
from scipy import spatial
import nltk
nltk.download('punkt')
import os

def get_ten_most_related(wikipedia_title):
  return 

def get_identifiers():
  # loop to find all tokens and counts
  unique_tokens = []
  document_titles = []
  for root, dirs, files in os.walk("corpus"):
    for current_dir in dirs:
      path1 = os.path.join(root, current_dir)
      for root2, dirs2, files2 in os.walk(path1):
        for current_file in files2:
          print("Find all unique tokens:", current_file)
          path2 = os.path.join(root2, current_file)
          current_file_text = open(path2, "r").read()
          tokens = nltk.tokenize.word_tokenize(current_file_text)
          # -5 to remove \n.txt for title
          document_titles.append(current_file[:-5])
          for token in tokens:
            if token not in unique_tokens:
              unique_tokens.append(token)

  print(unique_tokens[:10])
  print(document_titles[:10])

  # stores token identifiers in token_ids and document identifiers in doc_ids to be used to fill doc_term_matrix, runs in O(n)
  identifiers_tokens = {}
  index = 0
  for token in unique_tokens:
    identifiers_tokens[token] = index
    index += 1

  identifiers_docs = {}
  index = 0
  for doc in document_titles:
    identifiers_docs[doc] = index
    index +=1

  return identifiers_tokens, identifiers_docs

def bag_of_words(token_ids, doc_ids):
  """ Returns a populated document-term numpy matrix.

  token_ids -- dictionary mapping unique tokens to an id (column)
  doc_ids -- dictionary mapping document to id (row)
  """
  doc_term_matrix = np.zeros((len(token_ids), len(doc_ids)))

  print("Now populating doc_term_matrix")
  for root, dirs, files in os.walk("corpus"):
    for current_dir in dirs:
      path1 = os.path.join(root, current_dir)
      for root2, dirs2, files2 in os.walk(path1):
        for current_file in files2:
          print("Construct BOW:", current_file)
          path2 = os.path.join(root2, current_file)
          current_file_text = open(path2, "r").read()
          tokens = nltk.tokenize.word_tokenize(current_file_text)
          # -5 to remove \n.txt for title
          doc_name = current_file[:-5]
          for token in tokens:
            #doc_term_matrix[unique_tokens.index(token)][document_titles.index(doc_name)] += 1
            doc_term_matrix[token_ids[token]][doc_ids[doc_name]] += 1
  return doc_term_matrix

def get_cosine_similarities():
  return


tokens, docs = get_identifiers()
BOW = bag_of_words(tokens, docs)
print(BOW[:5][:4])
np.save("component1data", BOW)