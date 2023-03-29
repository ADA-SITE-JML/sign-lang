import random
import requests

# ✅ Random word generator using local file system

''''
def get_list_of_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()


words = get_list_of_words('random_words.txt')
random_word = random.choice(words)
print(random_word)  # 👉️ sales
'''
# --------------------------------------------------

# ✅ Random word generator using remote database
#SECOND WAY

def get_list_of_words():
    response = requests.get(
        'https://www.mit.edu/~ecprice/wordlist.10000',
        timeout=10
    )

    string_of_words = response.content.decode('utf-8')

    list_of_words = string_of_words.splitlines()

    return list_of_words


words = get_list_of_words()


def get_word(loopy):
    return random.choice(words)



