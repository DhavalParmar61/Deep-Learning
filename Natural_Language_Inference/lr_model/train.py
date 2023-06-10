 from sklearn.feature_extraction.text import TfidfVectorizer
 from processing import get_data
 from config import 
 
 def run_training()-> None:
    [corpus_train, Y_train] = get_data('snli_1.0_train.jsonl')
    Y_train = GL_to_int(Y_train)

    corpus1_train = []
    corpus2_train = []
    for i in range(len(corpus_train)//2):
        corpus1_train.append(corpus_train[2*i])
        corpus2_train.append(corpus_train[2*i+1])

    vectorizer_s1 = TfidfVectorizer()
    s1_train = vectorizer_s1.fit_transform(corpus1_train)
    f1 = open('feature_s1.pkl','wb')
    pickle.dump(vectorizer_s1,f1)
    f1.close

    vectorizer_s2 = TfidfVectorizer()
    s2_train = vectorizer_s2.fit_transform(corpus2_train)
    f2 = open('feature_s2.pkl','wb')
    pickle.dump(vectorizer_s2,f2)
    f2.close

    X_train = hstack([s1_train,s2_train])

    # Classifier
    model = LogisticRegression(max_iter = 5000)
    model.fit(X_train, Y_train)
    pickle.dump(model,open('./models/LR_model.sav','wb'))
        
if __name__=="__main__":
    run_training()