import re
import string
from torch import clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import copy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# modelName = "indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained("./indobert-base-p1_tokenizer/")
model = AutoModel.from_pretrained("./indobert-base-p1_model/")

# tokenizer = AutoTokenizer.from_pretrained("./xlm-roberta-large_tokenizer/")
# model = AutoModel.from_pretrained("./xlm-roberta-large_model/")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword = set(stopwords.words("indonesian"))


def check_similarity(
    text,
    kamus,
    number1=0,
    punctuation1=1,
    lower1=0,
    stem1=1,
    stopword1=0,
    number2=0,
    punctuation2=1,
    lower2=0,
    stem2=1,
    stopword2=0,
    knn=-1,
    word1=[],
):
    teks = text
    kamus2 = copy.deepcopy(kamus)
    # kalimat = sent_tokenize(
    #     teks,
    #     # language='indonesian'
    # )  # split text into sentences
    for word3 in kamus.keys() if len(word1) == 0 else word1:
        for word4 in kamus[word3].keys():
            sentences = []
            sentences.append(kamus[word3][word4])

            # for k in kalimat:
            #     sentences.append(k)
            sentences.append(teks)

            for i in range(1):
                sentences[i] = (
                    re.sub(r"\d+", "", sentences[i]) if number1 == 1 else sentences[i]
                )  # remove numbers
                sentences[i] = (
                    sentences[i].translate(
                        str.maketrans(string.punctuation, " " * len(string.punctuation))
                    )
                    if punctuation1 == 1
                    else sentences[i]
                )  # remove punctuation
                sentences[i] = (
                    sentences[i].lower() if lower1 == 1 else sentences[i]
                )  # lower case
                sentences[i] = (
                    stemmer.stem(sentences[i]) if stem1 == 1 else sentences[i]
                )  # stemming

                if stopword1 == 1:
                    sentences[i] = word_tokenize(sentences[i])  # tokenization
                    sentences[i] = [
                        word for word in sentences[i] if not word in listStopword
                    ]  # remove stopwords
                    sentences[i] = " ".join(sentences[i])  # join words

                sentences[i] = sentences[
                    i
                ].strip()  # remove leading and trailing whitespace
                sentences[i] = re.sub(r"\s+", " ", sentences[i])  # remove extra space

            for i in range(1, len(sentences)):
                sentences[i] = (
                    re.sub(r"\d+", "", sentences[i]) if number2 == 1 else sentences[i]
                )
                sentences[i] = (
                    sentences[i].translate(
                        str.maketrans(string.punctuation, " " * len(string.punctuation))
                    )
                    if punctuation2 == 1
                    else sentences[i]
                )
                sentences[i] = sentences[i].lower() if lower2 == 1 else sentences[i]
                sentences[i] = (
                    stemmer.stem(sentences[i]) if stem2 == 1 else sentences[i]
                )

                if stopword2 == 1:
                    sentences[i] = word_tokenize(sentences[i])  # tokenization
                    sentences[i] = [
                        word for word in sentences[i] if not word in listStopword
                    ]  # remove stopwords
                    sentences[i] = " ".join(sentences[i])  # join words

                sentences[i] = sentences[
                    i
                ].strip()  # remove leading and trailing whitespace
                sentences[i] = re.sub(r"\s+", " ", sentences[i])  # remove extra space
            # print(sentences)

            # initialize dictionary to store tokenized sentences
            tokens = {"input_ids": [], "attention_mask": []}

            for sentence in sentences:
                # encode each sentence and append to dictionary
                new_tokens = tokenizer.encode_plus(
                    sentence,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokens["input_ids"].append(new_tokens["input_ids"][0])
                tokens["attention_mask"].append(new_tokens["attention_mask"][0])

            # reformat list of tensors into single tensor
            tokens["input_ids"] = torch.stack(tokens["input_ids"])
            tokens["attention_mask"] = torch.stack(tokens["attention_mask"])
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / summed_mask

            # convert from PyTorch tensor to numpy array
            mean_pooled = mean_pooled.detach().numpy()

            # calculate
            x = cosine_similarity([mean_pooled[0]], mean_pooled[1:])

            # print("kompetensi: " + word1 + ", level: " + word2)

            x = x[0]
            # print(x)
            x.sort()
            # print(x[-knn:])
            if len(x) < knn or knn < 1:
                result = sum(x) / len(x)
            else:
                result = sum(x[-knn:]) / len(x[-knn:])
            # print(result)
            kamus2[word3][word4] = round(result,3)
    return kamus2

# res = check_similarity(teks, kamus)
