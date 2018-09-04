
# coding: utf-8

abuses = []
with open('bad_words.txt') as f:
    for word in f:
        word = word.strip('\n')
        abuses.append(word.lower())


Emoticons = ['\U0001f602', '\U0001f620', '\U0001f1ee', '\U0001f1f3', '\U0001f3c2', '\U0001f31f', '\U0001f3fb', '\U0001f604', '\U0001f64f', '\U0001f61c', '\U0001f60a', '\U0001f4af', '\U0001f518', '\U0001f44b', '\U0001f590']
for emo in Emoticons:
    print(len(str(emo)))


import string
f = open('processed_data_withoutID.txt', 'r')
text = f.readlines()



count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

Emoticons = ["\U0001f602", "\U0001f620", "\U0001f1ee", "\U0001f1f3", "\U0001f3c2", "\U0001f31f", "\U0001f3fb", "\U0001f604","\U0001f64f", "\U0001f61c", "\U0001f60a", "\U0001f4af", "\U0001f518", "\U0001f44b", "\U0001f590"]

punct = []
upper = []
phone = []
urlBin = []
sentLen = []
avgWordLen = []
singleLetters = []
url = 0
hashTags = []

f1 = open('featureSet.tsv', 'w+')
f1.write("text"+"\t"+"AggWord"+"\t"+"totalPunct"+"\t"+"hashCount"+"\t"+"totUpper"+"\t"+"phoneNum"+"\t"+"haveURL"+"\t"+"totalTok"+"\t"+"UniCount"+"\t"+"avgWordSize"+"\t"+"SentLen"+"\t"+"tag"+"\n")
for line in text:
    line = line.strip('\n').split('\t')
    tag = line[1]
    line = line[0]
    
    emoti = 0
    line = str(line)
    for em in Emoticons:
        if em in line.split(" "):
            print(em)
        
    
    agg=0
    for word in line.split(" "):
        if word.lower() in abuses:
            agg+=1
            
    punct_count = count(line, string.punctuation)
    upper_count = sum(1 for c in line if c.isupper())
    phone_count = line.count('+9')
    if('http:' in line):
        urlBin.append(1)
        url = 1
    else:
        urlBin.append(0)
        url = 0
    #print(punct_count)
    hash_count = sum(1 for c in line if c == '#')
    punct.append(punct_count)
    upper.append(upper_count)
    phone.append(phone_count)
    
    words = line.split(" ")
    sentLen.append(len(words))
    
    uniCount = sum(1 for x in words if len(x)==1)
    singleLetters.append(uniCount)
    
    tot = sum(len(x) for x in words)
    avgWordLen.append(int(tot/len(words)))
    str1 = (line+"\t"+str(agg)+"\t"+str(punct_count)+"\t"+str(hash_count)+"\t"+str(upper_count)+"\t"+str(phone_count)+"\t"+str(url)+"\t"+str(len(words))+"\t"+str(uniCount)+"\t"+str(int(tot/len(words)))+"\t"+str(len(line))+"\t"+tag)
    f1.write(str1+"\n")

    
f1.close()
# print(punct)
# print(upper)
# print(phone)
# print(urlBin)




#print(len(phone))

#print(sentLen)

#print(avgWordLen)

#print(singleLetters)

