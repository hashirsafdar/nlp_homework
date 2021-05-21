import wikipedia

# write wikipedia titles to individual files
titles = open("corpus/movies.txt", "r")
count = 1
for row in titles:
  try:
    article_text = wikipedia.page(row).content
    output_file = open("corpus/movies/" + row + ".txt", "w")
    output_file.write(article_text)
    print(count)
    count = count+1
  except:
    continue
