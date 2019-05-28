import json
import string
import math
import _pickle as pickle
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from pymongo import MongoClient

BOOK_KEEPING = "WEBPAGES_RAW_TEST/bookkeeping.json"
FILE_URL_PAIRS = dict()

INVERTED_INDEX = defaultdict()
DOC_TERM_COUNT = defaultdict()

INVERTED_INDEX_JSON = "INVERTED_INDEX.JSON"
DOC_TERM_COUNT_JSON = "DOC_TERM_COUNT.JSON"

INVERTED_INDEX_PICKLE = "INVERTED_INDEX.pickle"
DOC_TERM_COUNT_PICKLE = "DOC_TERM_COUNT.pickle"

DEBUG = False

# nltk.download("stopwords")
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def map_file_doc():
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


def create_index(doc, text):
    """Creates inverted index with positional information and adds documents to document set for counting later."""
    # TODO: Break each word into high and low lists
    # TODO: OR sort by docs in each word by tf-idf
    # TODO: Figure out better heuristic, possibly by analyzing HTML tags

    term_count = 0
    for index, word in enumerate(word_tokenize(text)):
        term_count += 1
        if word not in STOP_WORDS:
            word = STEMMER.stem(word)
            if word not in INVERTED_INDEX:
                INVERTED_INDEX[word] = {doc: {"locations": [index]}}
            else:
                if doc in INVERTED_INDEX[word]:
                    INVERTED_INDEX[word][doc]["locations"].append(index)
                else:
                    INVERTED_INDEX[word][doc] = {"locations": [index]}
    if doc not in DOC_TERM_COUNT:
        DOC_TERM_COUNT[doc] = term_count


def add_tf_idf():
    for term in INVERTED_INDEX:
        try:
            for doc in INVERTED_INDEX[term]:
                INVERTED_INDEX[term][doc]["tf-idf"] = calc_tf_idf(term, doc, INVERTED_INDEX, DOC_TERM_COUNT)
        except TypeError:
            pass


def calc_tf(term, doc, inverted_index, doc_term_count):
    return len(inverted_index[term][doc]) / doc_term_count[doc]


def calc_idf(term, inverted_index):
    return math.log((inverted_index["doc_count"] / len(inverted_index[term])))


def calc_tf_idf(term, doc, inverted_index, doc_term_count):
    """Calculate the tf-idf for a term and doc pair."""
    return calc_tf(term, doc, inverted_index, doc_term_count) * calc_idf(term, inverted_index)


def query_db(query):
    """Calculates the cosine similarity for query and docs, returns the highest 10"""
    result_all = {}
    result_top = []
    query = set(query.lower().strip().split())

    if DEBUG:
        with open(INVERTED_INDEX_JSON, "r") as file, open(DOC_TERM_COUNT_JSON, "r") as file2:
            inverted_index = json.load(file)
            doc_term_count = json.load(file2)
    else:
        with open(INVERTED_INDEX_PICKLE, "rb") as file, open(DOC_TERM_COUNT_PICKLE, "rb") as file2:
            inverted_index = pickle.load(file)
            doc_term_count = pickle.load(file2)

    for term in query:
        if term not in STOP_WORDS:
            term = STEMMER.stem(term)
            if term in inverted_index:
                for doc in inverted_index[term]:
                    if doc not in result_all:
                        result_all[doc] = 0
                    result_all[doc] += \
                        calc_tf_idf(term, doc, inverted_index, doc_term_count) * inverted_index[term][doc]["tf-idf"]
    if result_all:
        for r in sorted(result_all.items(), key=lambda item: -item[1]):
            result_top.append(r)
            if len(result_top) == 10:
                return result_top
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
    """Dumps in-memory index to a JSON or pickle file."""
    if DEBUG:
        with open(INVERTED_INDEX_JSON, "w") as inverted_index_file, open(DOC_TERM_COUNT_JSON, "w") as doc_term_count_file:
            json.dump(INVERTED_INDEX, inverted_index_file, indent=4)
            json.dump(DOC_TERM_COUNT, doc_term_count_file, indent=4)
    else:
        with open(INVERTED_INDEX_PICKLE, "wb") as inverted_index_file, open(DOC_TERM_COUNT_PICKLE, "wb") as doc_term_count_file:
            pickle.dump(INVERTED_INDEX, inverted_index_file)
            pickle.dump(DOC_TERM_COUNT, doc_term_count_file)


if __name__ == "__main__":
    if DEBUG:
        out = Path(INVERTED_INDEX_JSON)
    else:
        out = Path(INVERTED_INDEX_PICKLE)

    if not out.is_file():
        map_file_doc()
        for doc_, path in FILE_URL_PAIRS.items():
            processed = process_file(path)
            create_index(doc_, processed)
        INVERTED_INDEX["doc_count"] = len(DOC_TERM_COUNT)
        add_tf_idf()
        dump()

    search = input("Search: ").strip().lower()
    full_results = query_db(search)
    print_results(full_results)
