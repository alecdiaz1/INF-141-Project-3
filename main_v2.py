import json
import string
import math
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from pymongo import MongoClient

BOOK_KEEPING = "WEBPAGES_RAW_TEST/bookkeeping.json"
FILE_URL_PAIRS = dict()

INDEX = defaultdict()
DOCUMENTS = set()

# nltk.download("stopwords")
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def map_file_url():
    """Maps file names in bookkeeping.json to file paths."""
    with open(BOOK_KEEPING, encoding="UTF-8") as json_file:
        data = json.load(json_file)
        for line in data.items():
            folder_file_pair = line[0].split("/")
            abs_path2 = Path("WEBPAGES_RAW/") / Path(folder_file_pair[0], folder_file_pair[1])
            FILE_URL_PAIRS[line[1]] = abs_path2.absolute()  # URL is key, associated file path is value


def process_file(file_path):
    """Given a file path, returns the text of file in lowercase and stripped of punctuation."""
    with open(file_path, "r", encoding="UTF-8") as file:
        raw_html = file.read()
        soup = BeautifulSoup(raw_html, features="lxml")
        for script in soup(["script", "style"]):
            script.extract()
        # https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
        text = soup.get_text().lower()  # Make text lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Strip all punctuation before processing
    return text


def create_index(url, text):
    """Creates inverted index with positional information and adds documents to document set for counting later."""
    # TODO: Break each word into high and low lists
    # TODO: OR sort by urls in each word by tf-idf
    # TODO: Figure out better heuristic, possibly by analyzing HTML tags

    term_count = 0
    for index, word in enumerate(word_tokenize(text)):
        term_count += 1
        if word not in STOP_WORDS:
            word = STEMMER.stem(word)
            if word not in INDEX:
                INDEX[word] = {url: {"locations": [index]}}
            else:
                if url in INDEX[word]:
                    INDEX[word][url]["locations"].append(index)
                else:
                    INDEX[word][url] = {"locations": [index]}
            INDEX[word][url]["term_count"] = term_count
    DOCUMENTS.add(url)


def add_tf_idf():
    for term in INDEX:
        try:
            for url in INDEX[term]:
                INDEX[term][url]["tf-idf"] = calc_tf_idf(term, url, INDEX)
        except TypeError:
            pass


def calc_tf(term, url, db):
    return len(db[term][url]) / db[term][url]["term_count"]


def calc_idf(term, db):
    return math.log((db["doc_count"] / len(db[term])))


def calc_tf_idf(term, url, db):
    """Calculate the tf-idf for a term and url pair."""
    return calc_tf(term, url, db) * calc_idf(term, db)


def query_db(query):
    """Calculates the cosine similarity for query and urls, returns the highest 10"""
    result_all = {}
    result_top = []
    query = set(query.split())
    for term in query:
        if term not in STOP_WORDS:
            term = STEMMER.stem(term)
            with open("out.json", "r") as file:
                db = json.load(file)
                if term in db:
                    for url in db[term]:
                        if url not in result_all:
                            result_all[url] = 0
                        result_all[url] += calc_tf_idf(term, url, db) * db[term][url]["tf-idf"]
    if result_all:
        for r in sorted(result_all.items(), key=lambda item: -item[1]):
            result_top.append(r)
            if len(result_top) == 10:
                return result_top
    else:
        return None


def print_results(results):
    """Prints results, if there any. Otherwise prints no results found"""
    if results:
        for r in results:
            print(r)
    else:
        print("No results found")


def dump():
    """Dumps in-memory index to a JSON file."""
    with open("out.json", "w") as out:
        json.dump(INDEX, out, indent=4)


if __name__ == "__main__":
    out = Path("out.json")
    if not out.is_file():
        map_file_url()
        for url_, path in FILE_URL_PAIRS.items():
            processed = process_file(path)
            create_index(url_, processed)
        INDEX["doc_count"] = len(DOCUMENTS)
        add_tf_idf()
        dump()

    search = input("Search: ").strip().lower()
    full_results = query_db(search)
    print_results(full_results)
