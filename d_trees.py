import streamlit as st
import pandas as pd
from preprocessing import PreprocessData
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

@st.cache_data
def load_ner(option):
    if option == "mtsampled_ner_d4data":
        tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
        model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
    else:
        tokenizer = AutoTokenizer.from_pretrained("medical-ner-proj/albert-medical-ner-proj")
        model = AutoModelForTokenClassification.from_pretrained("medical-ner-proj/albert-medical-ner-proj")

    return model,tokenizer

def get_entities(text, model, tokenizer):
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="simple")
    ner_results = nlp_ner(text)
    words = []
    treatment = []
    test = []
    start= None
    end= None
    last_category = None
    
    for entity in ner_results:
        if entity["score"] > 0.5 and entity['entity_group'] in ['B_problem','I_problem','B_treatment','I_treatment','B_test','I_test']:
            category = entity['entity_group']
            if category.startswith('B_'):
                if start != None:
                    words.append(text[start:end].lower())
                    if last_category in ['B_treatment','I_treatment']:
                        treatment.append(text[start:end].lower())
                    if last_category in ['B_test','I_test']:
                        test.append(text[start:end].lower())
                start = entity['start']
                end = entity['end']
            else:
                end = entity['end']
            last_category = category
    
    if end != None:
        words.append(text[start:end].lower())
        if last_category in ['B_treatment','I_treatment']:
            treatment.append(text[start:end].lower())
        if last_category in ['B_test','I_test']:
            test.append(text[start:end].lower())

    return words,treatment,test


def parse_problem(text,model,tokenizer):
    tokens,treatment,test = get_entities(text, model, tokenizer)
    treatment = set(treatment)
    test = set(test)
    tokens = set(tokens)
    
    return tokens.difference(treatment).difference(test)

def get_recommendation(ml_model,problem,ylb):
    predicted =  ml_model.predict(problem)[0]
    return [ylb.classes_[i] for i,l in enumerate(predicted) if l]


def get_justification(br_clf,problem,xlb, ylb):
    predicted =  br_clf.predict(problem)[0]
    treatment =  [ylb.classes_[i] for i,l in enumerate(predicted) if l]
    clfs = [br_clf.estimators_[i] for i,l in enumerate(predicted) if l]
    justification = set()
    just_dict = dict()
    for clf,t in zip(clfs,treatment):
        temp = get_positive_decision_path(problem,clf,xlb.classes_)
        just_dict[t] = temp
        justification = justification.union(temp)

    return justification,just_dict

def get_positive_decision_path(sample,clf,features):
    node_indicator = clf.decision_path(sample)
    leaf_id = clf.apply(sample)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]
    
    words = set()
    
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[0] == node_id:
            continue
        if sample[0][feature[node_id]] > threshold[node_id]:
            words.add(features[feature[node_id]])
    return words

@st.cache_resource
def train_model(filename):
    df = PreprocessData(X_threshold=2, y_threshold=10,filename=f"data/{filename}.csv").get_preprocessed_data()
    # Convert the data to the format expected by the Binary Relevance class
    xlb = MultiLabelBinarizer()
    X = df['problem']
    X = xlb.fit_transform(X)

    ylb = MultiLabelBinarizer()
    y = df['treatment']
    y = ylb.fit_transform(y)

    br_clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=7))
    br_clf.fit(X, y)

    return br_clf,xlb,ylb

def render():
    st.markdown('## Treatment Recommendation')
    st.caption('Treatment recommendation with decision trees and decision paths as justification')

    option = st.selectbox(
    'Select a NER model',
    ('mtsampled_ner_2', 'mtsampled_ner_scibert', 'mtsampled_ner_d4data'),
    index=0)

    data_load_state = st.text('Loading data...')
    model,tokenizer = load_ner(option)
    ml_model,xlb,ylb = train_model(option)
    data_load_state.text(f'Loading data...done!')

    # Input field for the user to enter text
    text = st.text_area("Enter your complaint/problem here:", "")

    # Logic to process the input and generate output
    if text:
        problem = parse_problem(text,model,tokenizer)
        st.markdown("#### Extracted problems")
        st.caption('Problems are extracted using albert-medical-ner-proj NER (may not work well with other rule sets)')
        st.markdown(f"{' | '.join(problem)}")

        st.markdown("#### Recommended Treatment")
        treatment = get_recommendation(ml_model,xlb.transform([problem]),ylb)
        just, just_dict = get_justification(ml_model,xlb.transform([problem]),xlb,ylb)

        col1, col2, col3, col4 = st.columns(4)
        st.markdown("#### Justification")
        st.caption('Justifications are just the antecedents collected from treatment recommendation rules.')
        justification = st.text("Click the treatment to see justification.")

        for i,t in enumerate(treatment):
            if i % 4 == 0:
                with col1:
                    if st.button(t):
                        justification.text(f"{just_dict[t]}")

            if i % 4 == 1:
                with col2:
                    if st.button(t):
                        justification.text(f"{just_dict[t]}")

            if i % 4 == 2:
                with col3:
                    if st.button(t):
                        justification.text(f"{just_dict[t]}")

            if i % 4 == 3:
                with col4:
                    if st.button(t):
                        justification.text(f"{just_dict[t]}")