terms = {}
consonants = ["p", "b", "t", "d", "k", "ɡ", "t͡ʃ", "d͡ʒ", "s", "z", "ʃ", "ʒ", "h", "f", "v", "m", "n", "ŋ", "l", "r", "j"]
vowels = ["i", "y", "u", "e", "ø", "o", "æ", "ɑ"]

import csv
with open('karelian_terms.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        terms[row[0].lower()] = row[1].lower()

def mappings(word):
    mappings = {"ä": "æ", "ö": "ø", "š": "ʃ", "č": "t͡ʃ", "ž": "ʒ", "v": "ʋ", "ʹ": "ʲ", "a": "ɑ"}
    word = list(word)
    for i, char in enumerate(word):
        if char in mappings.keys():
            word[i] = mappings[char]
    word = ''.join(word)
    return word

def palatilization(word): #consonants in front of front vowels get palatilized
    word = list(word)
    for i in range(len(word)-1): #checking two letter segments
        if word[i] in consonants and word[i] not in ["h", "k", "j", "b", "r", "ʃ", "p", "m"]:
            if word[i+1] in ["i", "y", "e", "ø", "æ"]:
                word.insert(i+1, "ʲ")
    word = ''.join(word)
    return word

def gemination(word):
    word = list(word)
    for i in range(len(word)-1):
        if word[i] == word[i+1]:
            word[i+1] = "ː"
    word = ''.join(word)
    return word

def desyllabification(word): #second vowels in diphthongs get desyllabified
    desyllabified = {"i": "i̯", "y": "y̯", "u": "u̯", "e": "e̯", "ø": "ø̯", "o": "o̯", "æ": "æ̯", "ɑ": "ɑ̯"}
    word = list(word)
    for i in range(len(word)-1):
        if word[i] in vowels and word[i+1] in vowels:
            word[i+1] = desyllabified[word[i+1]]
    word = ''.join(word)
    return word

def apply_stress(word):
    word = list(word)
    word.insert(0, "ˈ")
    word = "".join(word)
    return word

def velarization(word):
    word = list(word)
    for i in range(len(word)-1):
        if word[i] == "n" and word[i+1] == "g":
            word[i] = "ŋ"
    word = ''.join(word)
    return word

def min_edit_distance(str1: str, str2: str) -> int:
    str1 = list(str1)
    str2 = list(str2)

    str1.pop(0)
    str1.pop(len(str1)-1)

    str2.pop(0)
    str2.pop(len(str2)-1)

    str1 = ''.join(str1)
    str2 = ''.join(str2)

    m, n = len(str1), len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j  
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],   
                    dp[i][j - 1],  
                    dp[i - 1][j - 1] 
                )
                
    return (dp[m][n]/m)

def g2p(word):
    word = apply_stress(word)
    word = gemination(word)
    word = mappings(word)
    word = palatilization(word)
    word = desyllabification(word)
    word = velarization(word)
    return(f"/{word}/")

transcribed = []
pers_list = []
for word in terms.keys():
    transcribed_term = g2p(word)
    transcribed.append(transcribed_term)
    per = min_edit_distance(transcribed_term, terms[word])
    pers_list.append([per, [transcribed_term, terms[word], word]])

avg = 0
for per in pers_list:
    avg += per[0]
avg /= len(pers_list)

sorted_pers = sorted(pers_list, reverse=False)
for i in range(len(pers_list)):
    print(sorted_pers[i])

#print(transcribed)
#print(pers)
print(avg*100)
#print(g2p("viizi"))
#print(min_edit_distance("/ˈʋiːzʲi/", "/ˈʋiːzʲi/"))

with open('karelian_output.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["orthography", "gold", "prediction", "accuracy"])
    for per in sorted_pers:
        spamwriter.writerow([per[1][2], per[1][1], per[1][0], round(per[0], 2)])
    

