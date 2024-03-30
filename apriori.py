import streamlit as st
import pandas as pd
from preprocessing import PreprocessData
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from common import *

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

def perform_apriori(transcription, min_support=1e-4, max_len=2):
    results = apriori(transcription, min_support=min_support, max_len=max_len, use_colnames=True, verbose=1, low_memory=True)
    return results

def get_recommendation(rules,problem):
    pass1 = pd.DataFrame()
    for p in problem:
        temp = rules[rules['antecedents'] == p]
        pass1 = pd.concat([pass1,temp])

    if len(pass1) == 0:
        return []

    pass1.drop_duplicates(inplace=True)
    pass1.sort_values(['confidence'],ascending=False,inplace=True)

    return list(set(pass1['consequents']))


def get_justification(rules,treatment,problem):
    pass1 = pd.DataFrame()
    just_dict = dict()
    for t in treatment:
        temp = rules[rules['consequents'] == t]
        just_dict[t] = set(temp['antecedents']).intersection(problem)

        pass1 = pd.concat([pass1,temp])

    if len(pass1) == 0:
        return [],just_dict
    
    pass1.drop_duplicates(inplace=True)
    pass1.sort_values(['confidence'],ascending=False,inplace=True)

    return list(set(pass1['consequents']).intersection(set(problem))),just_dict

@st.cache_data
def load_rules(filename):
    data = PreprocessData(X_threshold=2, y_threshold=10,filename=f"data/{filename}.csv").get_preprocessed_data()
    MAX_LENGTH = 2
    dataset = data['all_tokens'].apply(lambda x: list(x)).values
    # dataset = train.apply(lambda x: preprocess(x),axis=1).values
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = perform_apriori(df,min_support=1/data.shape[0],max_len=MAX_LENGTH)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules['antecedents'] = rules['antecedents'].apply( lambda x: list(x)[0].replace("frozenset({'","").replace("'})",""))
    rules['consequents'] = rules['consequents'].apply( lambda x: list(x)[0].replace("frozenset({'","").replace("'})",""))

    ylb = MultiLabelBinarizer()
    ylb.fit_transform( data['treatment'])
    all_treatments = ylb.classes_
    xlb = MultiLabelBinarizer()
    xlb.fit_transform( data['problem'])
    all_problems = xlb.classes_

    condition1 = rules['antecedents'].isin(all_problems) & rules['consequents'].isin(all_treatments)
    condition2 = rules['antecedents'].isin(all_treatments) & rules['consequents'].isin(all_problems)
    combined_condition = condition1 | condition2
    rules = rules[combined_condition]

    return rules

def render():
    st.markdown('## Treatment Recommendation')
    st.caption('Treatment recommendation with justification via apriori algorithm')

    option = st.selectbox(
    'Select a rule set',
    ('mtsampled_ner_2', 'mtsampled_ner_scibert', 'mtsampled_ner_d4data'),
    index=0)

    data_load_state = st.text('Loading data...')
    model,tokenizer = load_ner(option)
    rules = load_rules(option)
    data_load_state.text(f'Loading data...done! Working with {rules.shape[0]} rules.')

    if 'text_value' not in st.session_state:
        st.session_state.text_value = ""
    c = st.container()
    display_examples('text_value')
    text = c.text_area("Enter your complaint/problem here:", st.session_state['text_value'],height=400)

    # Logic to process the input and generate output
    if text:
        problem = parse_problem(text,model,tokenizer)
        st.markdown("#### Extracted problems")
        st.caption('Not yet implemented for SciBert (uses albert-medical-ner-proj instead)')
        st.markdown(f"{' | '.join(problem)}")

        st.markdown("#### Recommended Treatment")
        treatment = get_recommendation(rules,problem)
        just, just_dict = get_justification(rules,treatment,problem)

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