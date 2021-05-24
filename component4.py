from gensim.models import LsiModel
import numpy as np
import json


infile = open("word2id.json", 'r')
word2id = json.load(infile)
id2word = {int(k):v for v, k in word2id.items()}


saved = np.load("tf-idf.npy")
tf_idf = [[(i, x) for i, x in enumerate(e)] for e in saved]


print("Training model (this may take a while!)...")
model = LsiModel(tf_idf, id2word=id2word, num_topics=200)
document_lsa_vectors = [model[doc] for doc in tf_idf]
document_lsa_vectors = [np.array([component[1] for component in vector]) for vector in document_lsa_vectors]



# model_export_data = []
# for doc_id in sorted(document_ids):
#   document_vector = document_lsa_vectors[doc_id] 
#   document_topic = topics[doc_id] 
#   # TODO MAKE THIS WORK 
#     document_entry = (document_topic, document_vector) 
#     model_export_data.append(document_entry)
# # Convert to numpy array and save to file
# model_export_data_array = np.array(model_export_data)
# outfile = open('lsa.npy', 'wb')
# np.save(outfile, model_export_data_array)
# outfile.close()