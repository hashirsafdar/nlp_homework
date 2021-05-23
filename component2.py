import numpy as np
from scipy import spatial
import nltk
nltk.download('punkt')
import os


def get_identifiers():
  # # loop to find all tokens and counts
  # unique_tokens = []
  # document_titles = []
  # for root, dirs, files in os.walk("corpus"):
  #   for current_dir in dirs:
  #     path1 = os.path.join(root, current_dir)
  #     for root2, dirs2, files2 in os.walk(path1):
  #       for current_file in files2:
  #         print("Find all unique tokens:", current_file)
  #         path2 = os.path.join(root2, current_file)
  #         current_file_text = open(path2, "r").read()
  #         tokens = nltk.tokenize.word_tokenize(current_file_text)
  #         # -5 to remove \n.txt for title
  #         document_titles.append(current_file[:-5])
  #         for token in tokens:
  #           if token not in unique_tokens:
  #             unique_tokens.append(token)

  # print(unique_tokens[:10])
  # print(document_titles[:10])

  # np.save("unique_tokens", np.array(unique_tokens))
  # np.save("doc_titles", np.array(document_titles))
  unique_tokens = np.load("unique_tokens.npy")
  document_titles = np.load("doc_titles.npy")

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
  #doc_term_matrix = np.zeros((len(token_ids), len(doc_ids)))

  # print("Now populating doc_term_matrix")
  # for root, dirs, files in os.walk("corpus"):
  #   for current_dir in dirs:
  #     path1 = os.path.join(root, current_dir)
  #     for root2, dirs2, files2 in os.walk(path1):
  #       for current_file in files2:
  #         path2 = os.path.join(root2, current_file)
  #         with open(path2, "r") as currfile:
  #           current_file_text = currfile.read()
  #         tokens = nltk.tokenize.word_tokenize(current_file_text)
  #         # -5 to remove \n.txt for title
  #         doc_name = current_file[:-5]
  #         print("Construct BOW:", doc_name)
  #         for token in tokens:
  #           doc_term_matrix[token_ids[token]][doc_ids[doc_name]] += 1
  # return doc_term_matrix
  matrix = np.load("component1docterm.npz")
  return matrix['arr_0']


def get_ten_most_related(wikipedia_title, docs, doc_term_matrix):
  if wikipedia_title not in docs:
    print("invalid")
    return None

  document_titles = np.load("doc_titles.npy")

  index = docs[wikipedia_title]
  top_10 = [(0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,""), (0.0,"")]

  for i in range(1297):
    relatedness = 1 - spatial.distance.cosine(doc_term_matrix[index], doc_term_matrix[i])
    #skip completely similar article that is just itself
    if relatedness == 1.0:
      continue
    # print(relatedness)
    for j in range(len(top_10)):
      #code will replace the current top 10 element if 
      #relatedness score is less and there are no other 0s
      #for the related score otherwise it replaces the 0
      if relatedness > top_10[j][0] and (0.0,"") not in top_10:
        title = document_titles[i]
        print("this \n")
        print(relatedness)
        top_10[j] = (relatedness, title)
        #return
        break
      elif top_10[j] == (0.0,""):
        title = document_titles[i]
        print("that \n")
        print(relatedness)
        top_10[j] = (relatedness, title)  
        break

  #sorts list in descending order
  top_10.sort(key=lambda x: x[0], reverse=True)
  return top_10


def get_cosine_similarities(doc_term_matrix):
  """
  0-550 games
  551-870 war
  871-1296 movies
  """
  games_games = 0
  games_war = 0
  games_movies = 0
  war_war = 0
  war_movies = 0
  movies_movies = 0
  gg_count = 0
  for x in range(1297):
    print("Comparing document number " + str(x))
    for y in range(1297):
      relatedness = 1 - spatial.distance.cosine(doc_term_matrix[x], doc_term_matrix[y])
      if x <= 550 and y <= 550:
        gg_count += 1
        games_games += relatedness
      elif x <= 550 and y <= 870:
        games_war += relatedness
      elif x <=550 and y <= 1296:
        games_movies += relatedness
      elif x <= 870 and y <= 870:
        war_war += relatedness
      elif x <= 870 and y <= 1296:
        war_movies += relatedness
      elif x <= 1296 and y <= 1296: 
        movies_movies += relatedness
      else:
        print("BIG BOY ERROR \n")
        print("x " + str(x) + "\n")
        print("y " + str(y) + "\n")
    #if x == 551:
    #  print(x, gg_count)
    #  return
  '''
  1st -> 1295
  1296
  1295
  1294
  1293

  2nd -> 1294
  3rd -> 1293
  1293rd -> 2
  '''

  games_games = games_games / (551 * 551)
  games_war = games_war / (551 * 320)
  games_movies = games_movies / (551 * 426)
  war_war = war_war / (320 * 320)
  war_movies = war_movies / (320 * 426)
  movies_movies = movies_movies / (426 * 426)

  
  return [[games_games, games_war, games_movies],
          [0, war_war, war_movies],
          [0, 0, movies_movies]]


tokens, docs = get_identifiers()
BOW = bag_of_words(tokens, docs)
similarity = get_cosine_similarities(BOW)
for s in similarity:
  print(s)
#print(get_ten_most_related("American Chopper 2: Full Throttle", docs, BOW))

