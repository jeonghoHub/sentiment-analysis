import pandas as pd
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# 텍스트 전처리 함수: 숫자 제거
def preprocess_text(text):
    text = re.sub(r"\d+", " ", text)
    return text


# Okt를 사용한 텍스트 토큰화 함수
def tw_tokenizer(text):
    tokens_ko = Okt().morphs(text)
    return tokens_ko


# 학습 데이터 불러오기 및 전처리
train_df = pd.read_csv(r'data/ratings_train.txt', sep='\t')
train_df = train_df.fillna(' ')
train_df['document'] = train_df['document'].apply(preprocess_text)

# 테스트 데이터 불러오기 및 전처리
test_df = pd.read_csv(r'data/ratings_test.txt', sep='\t')
test_df = test_df.fillna(' ')
test_df['document'] = test_df['document'].apply(preprocess_text)

# TF-IDF 벡터라이저 생성
tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1, 2), min_df=3, max_df=0.9)

# 학습 데이터에 벡터라이저 적용
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])

# 로지스틱 회귀 분류기 생성
lg_clf = LogisticRegression(random_state=0)

# 그리드 서치를 위한 하이퍼파라미터 설정
params = {'C': [1, 3.5, 4.5, 5.5, 10]}

# 그리드 서치 및 교차 검증 수행
grid_cv = GridSearchCV(lg_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(tfidf_matrix_train, train_df['label'])

# 최적의 파라미터와 점수 출력
print(grid_cv.best_params_, grid_cv.best_score_)

# 테스트 데이터를 학습된 TF-IDF 벡터라이저로 변환
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])

# 그리드 서치에서 최적의 추정기 가져오기
best_estimator = grid_cv.best_estimator_

# 테스트 데이터에 대한 예측 수행
preds = best_estimator.predict(tfidf_matrix_test)

# 테스트 데이터에 대한 정확도 계산 및 출력
accuracy = accuracy_score(test_df['label'], preds)
print(f"Accuracy on test data: {accuracy:.2f}")

# 사용자 입력 받아 긍정적 또는 부정적으로 분류하는 루프
while True:
    user_input = input("Enter a Korean sentence (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_input = preprocess_text(user_input)
    user_tfidf = tfidf_vect.transform([user_input])
    prediction = best_estimator.predict(user_tfidf)

    if prediction[0] == 1:
        print("긍정적인 텍스트 입니다")
    else:
        print("부정적인 텍스트 입니다.")
