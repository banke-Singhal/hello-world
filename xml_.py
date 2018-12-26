import nltk

from nltk.tokenize import RegexpTokenizer



sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""

sentence = sentence.lower()
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(sentence)
print(tokens)

print(sentence)

tokens = nltk.word_tokenize(sentence)

print(tokens)


import string

print(string.punctuation)
sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""

sentence1 = sentence.split(' ')

def remove_pun(text):
    sentence1 = "".join([ i for i in text if i not in string.punctuation])
    return sentence1
    




sen = list(map(lambda x : remove_pun(x), sentence1))
print(sen)

sen11 =  [remove_pun(i) for i in sentence1]

print(sen11)


a = list([(lambda x : remove_pun(x)), sentence1])
print(type(a))
print(a[1])

print(a())

print(sen)



f = lambda x: x*x
t=[f(x) for x in range(10)]
print(t)

x = [(lambda x : remove_pun(x))(x) for x in sentence1]

print(x)



print(sentence1)


word = 'geeks, for, geeks, pawan'
  
# maxsplit: 0 
test1 = word.split(', ', 0)
print(test1)
print(len(test1))
print(type(test1)) 
  
# maxsplit: 4 
test2 = word.split(', ', 4)
print(test2[2:3])
print(len(test2))
print(type(test2)) 
  
# maxsplit: 1 
test3 = word.split(', ', 1)
print(test3)
print(len(test3))
print(type(test3)) 


test1 = map(lambda x :x*2 , [1,2,3,4])
print(map(lambda x : x*2, [1, 2, 3, 4]))
print(test1)




dict_a = [{'name': 'python', 'points': 10}, {'name': 'java', 'points': 8}]
  
list(map(lambda x : x['name'], dict_a) )# Output: ['python', 'java']
  
map(lambda x : x['points']*10,  dict_a) # Output: [100, 80]

map(lambda x : x['name'] == "python", dict_a) # Output: [True, False]


a = [1,2,3,4,5,6]
b = filter(lambda x : x %2 ==0 ,a)

print(b)

dict_a = [{'name': 'python', 'points': 10}, {'name': 'java', 'points': 8}]
test = dict(filter(lambda x : x['name']=='python' , dict_a ))
print(test)

from functools import reduce
product = reduce(lambda x, y: x  y, [1, 2, 3, 4])
print(product)





from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 
data = ["All work and no play makes jack dull boy chancess. All work and no play makes jack a dull boy."]
data1 = word_tokenize(data)
print(data1)

swords = stopwords.words('english')
print(swords)

data2 = [ i for i in data1 if i not in swords ]
print(data2)
ps = nltk.PorterStemmer()

data3 = [ps.stem(i) for i in data2]

print(data3)

wn = nltk.WordNetLemmatizer()
data4 = [wn.lemmatize(i) for i in data2]
print(data4)


lemmatizer = nltk.WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(data)
print(count_vect.vocabulary_)
vector = count_vect.transform(data)
print(count_vect.get_feature_names())
print(vector.shape)
print(type(vector))
print(vector.toarray())






from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector)
print(vector.shape)
print(type(vector))
print(vector.toarray())
print(vectorizer.vocabulary_.get("the"))



from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize0
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
print(vector)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

