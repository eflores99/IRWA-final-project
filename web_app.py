import nltk
from flask import Flask, render_template
from flask import request
import os
print("current dir", os.getcwd() + "\n")

from myapp.analytics.analytics_data import AnalyticsData, Click
from myapp.search_engine.search_engine import SearchEngine, cleanText
from myapp.search_engine.text_processing import all_tweets
from myapp.core.utils import load_documents_corpus

#from analytics import AnalyticsData, Click
#import utils 

app = Flask(__name__)

corpus = load_documents_corpus()
print("File loaded")
# Putting the preprocessed text back into the dictionary
for i in corpus.keys():
    corpus[i]['full_text'] = cleanText(corpus[i]['full_text'])
print("Cleaning completed")
#Creating the list of preprocessed tweets
clean_terms = []
for i in corpus.keys():
     clean_terms.append(corpus[i]['full_text'])
print("Corpus loaded")
searchEngine = SearchEngine()
print("Start creating indices")
searchEngine.create_index(clean_terms,corpus)
print("Indices created")
analytics_data = AnalyticsData()

@app.route('/')
def search_form():
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    results = searchEngine.search(search_query,corpus)
    found_count = len(results)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')
    clicked_doc_id = int(request.args["id"])
    analytics_data.fact_clicks.append(Click(clicked_doc_id, "some desc"))

    print("click in id={} - fact_clicks len: {}".format(clicked_doc_id, len(analytics_data.fact_clicks)))

    return render_template('doc_details.html')


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    ### Start replace with your code ###
    visited_docs = []
    for clk in analytics_data.fact_clicks:
        visited_docs.append((corpus[clk.doc_id]))

    return render_template('stats.html', clicks_data=visited_docs)
    ### End replace with your code ###

@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port="8088", host="0.0.0.0", threaded=False, debug=True)
