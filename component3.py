from scipy import spatial
import nltk
import json
nltk.download('punkt')
nltk.download('wordnet')
import os
import string
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer


def save_identifiers():
  # loop to find all tokens and counts
  unique_tokens = []
  # document_titles = []
  for root, dirs, files in os.walk("comp3_final_corpus"):
    for current_file in files:
      print("Find all unique tokens:", current_file)
      path2 = os.path.join(root, current_file)
      current_file_text = open(path2, "r").read()
      tokens = nltk.tokenize.word_tokenize(current_file_text)
      # # -5 to remove \n.txt for title
      # document_titles.append(current_file[:-5])
      for token in tokens:
        if token not in unique_tokens:
          unique_tokens.append(token)

  # word2id = {}
  # for i in range(len(unique_tokens)):
  #   word2id[unique_tokens[i]] = i
  # with open("word2id", "w") as json_file:
  #   json.dump(word2id, json_file)

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


  # with open("id2word", "w") as json_file:
  #   json.dump(identifiers_tokens, json_file)

  return identifiers_tokens, identifiers_docs

def load_tokens():
  infile = open("word2id.json", 'r')
  loaded_word2id = json.load(infile)
  #loaded_id2word = {v:int(k): v for k, v in loaded_id2word.items()}

  document_titles = np.load("doc_titles.npy")
  identifiers_docs = {}
  index = 0
  for doc in document_titles:
    identifiers_docs[doc] = index
    index +=1

  return loaded_word2id, identifiers_docs

def bag_of_words(token_ids, doc_ids):
  """ Returns a populated document-term numpy matrix.

  token_ids -- dictionary mapping unique tokens to an id (column)
  doc_ids -- dictionary mapping document to id (row)
  """
  num_docs = len(doc_ids)
  num_tokens = len(token_ids)
  doc_term_matrix = np.zeros((num_docs, num_tokens))

  print("Now populating doc_term_matrix")
  for root, dirs, files in os.walk("comp3_final_corpus"):
    for current_file in files:
      path2 = os.path.join(root, current_file)
      with open(path2, "r") as currfile:
        current_file_text = currfile.read()
      tokens = nltk.tokenize.word_tokenize(current_file_text)
      # -5 to remove \n.txt for title
      doc_name = current_file[:-5]
      print("Construct BOW:", doc_name)
      for token in tokens:
        doc_term_matrix[doc_ids[doc_name]][token_ids[token]] += 1

  # convert standard BOW to tfidf
  tf_idf = np.zeros((num_docs, num_tokens))

  for i in range(num_docs):
    doc_word_count = doc_term_matrix.sum(axis = 1)[i]
    total_word_count = doc_term_matrix.sum(axis = 0)
    print("Construct tf-idf for doc number " + str(i) + " with word count " + str(doc_word_count))
    for j in range(num_tokens):
      tf = doc_term_matrix[i][j] / doc_word_count
      idf = np.log10(1297/total_word_count[j])
  
      tf_idf[i][j] = tf*idf

  return tf_idf


def get_ten_most_related(wikipedia_title, docs, doc_term_matrix):
  if wikipedia_title not in docs:
    print("invalid")
    return

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
  for x in range(1296):
    print("Comparing document number " + str(x))
    for y in range(1296):
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
        print("ERROR \n")
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
  games_war = games_war / (551 * 319)
  games_movies = games_movies / (551 * 425)
  war_war = war_war / (319 * 319)
  war_movies = war_movies / (319 * 425)
  movies_movies = movies_movies / (425 * 425)

  
  return [[games_games, games_war, games_movies],
          [0, war_war, war_movies],
          [0, 0, movies_movies]]

def create_component_3_corpus():
  words_that_occur_in_one_doc = []
  all_words_that_occur = []
  lemmatizer = WordNetLemmatizer()
  stopwords = open("stopwords.txt", "r").readlines()
  stopwords = [word.strip('\n') for word in stopwords]
  for root, dirs, files in os.walk("corpus"):
    for current_dir in dirs:
      path1 = os.path.join(root, current_dir)
      for root2, dirs2, files2 in os.walk(path1):
        for current_file in files2:
          cleaned_file = open("component_3_corpus/" + current_file, "w")
          path2 = os.path.join(root2, current_file)
          current_file_text = open(path2, "r").read().lower()
          #print(type(current_file_text))
          for punct in string.punctuation:
            if punct == "'":
              continue
            current_file_text = current_file_text.replace(punct, f" {punct} ")
          current_file_text = current_file_text.split()
          output_file_tokens = []
          for token in current_file_text:
            if token not in string.punctuation:
              lemmatized_token = lemmatizer.lemmatize(token)
              if lemmatized_token not in stopwords:
                output_file_tokens.append(lemmatized_token)
          for token in output_file_tokens:
            print(token, file=cleaned_file, end=" ")
          #for word in current_file_text:
            #word_lemma = nltk.
          #tokens = nltk.tokenize.word_tokenize(current_file_text)
          # -5 to remove \n.txt for title
          #document_titles.append(current_file[:-4])
          #for token in tokens:
          #  if token not in unique_tokens:
          #    unique_tokens.append(token)

def remove_words_unique_to_document():
  unique_words = []
  all_words = {}
  #I think our doc-term matrix has been constructed incorrectly because it says the term Japan only occurs in one document
  #matrix = np.load("comp1BOW.npz")
  #matrix = matrix['arr_0']
  # unique_tokens = np.load("unique_tokens.npy")
  # for i in range(len(matrix)):
  #   column = []
  #   for j in range(len(matrix[0])):
  #     column.append(matrix[i][j])
  #   num_zeroes = column.count(0.0)
  #   if num_zeroes == len(matrix[0]) - 1:
  #     print(unique_tokens[i])
  #     unique_words.append(unique_tokens[i].lower())
  
  # print("Done", unique_words)
  # return
  for root, dirs, files in os.walk("component_3_corpus"):
    for current_file in files:
      path = os.path.join(root, current_file)
      text = open(path, "r")
      words = text.read()
      words = words.split()
      for word in words:
        if word in all_words.keys():
          if current_file not in all_words[word]:
            all_words[word].append(current_file[:-5])
        else:
          all_words[word] = []
          all_words[word].append(current_file)
  for root, dirs, files in os.walk("component_3_corpus"):
    for current_file in files:
      path = os.path.join(root, current_file)
      text = open(path, "r")
      words = text.read()
      words = words.split()
      for word in all_words.keys():
        if len(all_words[word]) == 1:
          while word in words:
            print("REMOVED:", word)
            words.remove(word)
      text.close()
      output_file = open(f"comp3_final_corpus/{current_file}", "w")
      print("Number of words:", len(words))
      for word in words:
        print(word, file=output_file, end=" ")
  return

def construct_tf_idf(token_ids, doc_ids):
  """ Returns a populated document-term numpy matrix populated with tf-idf values.

  token_ids -- dictionary mapping unique tokens to an id (column)
  doc_ids -- dictionary mapping document to id (row)
  """
  doc_term_matrix = np.zeros((len(token_ids), len(doc_ids)))
  all_words = {}
  print("Getting all_words")
  for root, dirs, files in os.walk("comp3_final_corpus"):
    for current_file in files:
      path = os.path.join(root, current_file)
      text = open(path, "r")
      words = text.read()
      words = words.split()
      for word in words:
        if word in all_words.keys():
          if current_file not in all_words[word]:
            all_words[word].append(current_file[:-5])
        else:
          all_words[word] = []
          all_words[word].append(current_file)

  print("Now populating doc_term_matrix")
  for root, dirs, files in os.walk("comp3_final_corpus"):
    for current_file in files:
      path2 = os.path.join(root, current_file)
      with open(path2, "r") as currfile:
        current_file_text = currfile.read()
      tokens = nltk.tokenize.word_tokenize(current_file_text)
      # -5 to remove \n.txt for title
      doc_name = current_file[:-5]
      print("Construct tf-idf:", doc_name)
      for token in tokens:
        try:
          tf_value = tokens.count(token) / len(tokens)
          idf_value = np.log10(len(files) / len(all_words[token]))
          doc_term_matrix[doc_ids[doc_name]][token_ids[token]] = tf_value*idf_value
        except:
          continue

  return doc_term_matrix


# tokens, docs = load_tokens()
# BOW = bag_of_words(tokens, docs)
# np.save("tf-idf", BOW)
# similarity = get_cosine_similarities(BOW)
# for s in similarity:
#   print(s)
# print(get_ten_most_related("Cars 2", docs, BOW))

#this creates a corpus using the files in the folder corpus but then lemmatizes
#all of it and also removes stopwords and punctuations.
#create_component_3_corpus()
remove_words_unique_to_document()
#print(get_cosine_similarities(BOW))
#np.save("component1data", BOW)

#get_identifiers()


