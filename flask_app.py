import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import nltk
import re
import numpy as np
from nltk.corpus import stopwords #stopwords are used to remve irrelevant words such as "is, to, so etc."
from nltk.stem.porter import PorterStemmer #Words are reduced to their root form such as "likes, liked, likely to be reduced to like"
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route("/", methods=["GET"]) #route for homepage
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def index():
    if request.method == "POST":
        try:
            message = request.form["msg"]
            filename = "final_spam_vs_ham_model.pickle"
            nltk.download("stopwords")
            ps = PorterStemmer()


            review = re.sub("[^a-zA-Z]", " ", message)
            review = review.lower()
            review = review.split()

            review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
            review = " ".join(review)
            corpus = [review]

            cv = CountVectorizer()
            corpus_list = []

            #To create same instance of CountVectorizer as of training model
            with open("corpus_list.txt", "r") as f:
                filecontents = f.readlines()
                for line in filecontents:
                    # To remove line break
                    current_point = line[:-1]
                    corpus_list.append(current_point)
            dummy_var = cv.fit_transform(corpus_list).toarray()

            x = cv.transform(corpus).toarray()


            loaded_model = pickle.load(open(filename, 'rb'))
            #prediction begins now
            prediction = loaded_model.predict(x)
            print("Prediction is: ", prediction[0])
            op = ""
            if prediction[0] == 0:
                op = "You just entered a HAM text"
            else:
                op = "You just entered a SPAM text"
            return render_template("results.html", result = op)
        except Exception as e:
            print("EXCEPTION OCCURED, PLEASE TRY IN A WHILE ",e)

    else:
        return render_template("index.html")

#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000) #For testing in local system
    #app.run(host="0.0.0.0", port=port)
    #app.run(host="127.0.0.3", debug=True)