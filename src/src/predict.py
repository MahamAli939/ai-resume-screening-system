import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_resume(text):

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)

    return prediction
