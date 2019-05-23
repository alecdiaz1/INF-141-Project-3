import json
import string
import math
from collections import defaultdict
from pathlib import Path
from nltk import word_tokenize

from bs4 import BeautifulSoup
from pymongo import MongoClient

BOOK_KEEPING = "WEBPAGES_RAW_TEST/bookkeeping.json"
FILE_URL_PAIRS = dict()

INDEX = defaultdict()
DOCUMENTS = set()


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
    for index, word in enumerate(word_tokenize(text)):
        if word not in INDEX:
            INDEX[word] = {url: [index]}
        else:
            if url in INDEX[word]:
                INDEX[word][url].append(index)
            else:
                INDEX[word].update({url: [index]})
        DOCUMENTS.add(url)


def calc_tf_idf(term, url, db):
    """Calculate the tf-idf for a term and url pair."""
    # TODO: Figure out better heuristic, possibly by analyzing HTML tags
    tf = len(db[term][url])
    idf = math.log((db["doc_count"] / len(db[term])), 10)
    return tf * idf


def query_db(term):
    """Calculates the tf-idf for every url the term occurs in. Returns the resulting urls and tf-idf, sorted."""
    result = {}
    with open("out.json", "r") as file:
        db = json.load(file)
        for url in db[term]:
            result[url] = calc_tf_idf(term, url, db)
    for r in sorted(result.items(), key=lambda item: -item[1]):
        print(r)
    # return sorted(result.items(), key=lambda item: -item[1])


def dump():
    """Dumps in-memory index to a JSON file."""
    with open("out.json", "w") as out:
        INDEX["doc_count"] = len(DOCUMENTS)
        json.dump(INDEX, out, indent=4)


if __name__ == "__main__":
    map_file_url()
    query = input("Search: ").strip().lower()
    query_db(query)

    # ----- RUN THIS ONLY IF YOU HAVE NO OUT.JSON FILE, COMMENT OUT AFTER -----
    # for url_, path in FILE_URL_PAIRS.items():
    #     processed = process_file(path)
    #     create_index(url_, processed)
    # dump()

