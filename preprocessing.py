import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer

class PreprocessData:
    def __init__(self, X_threshold=10, y_threshold=10, filename='mtsampled_ner_2.csv'):
        self.X_threshold = X_threshold
        self.y_threshold = y_threshold
        self.filename = filename

    def load_data(self):
        df = pd.read_csv(self.filename)
        if 'test' in df.columns:
            del df['test']
        print("Original Shape:", df.shape)
        return df

    def drop_duplicates(self, df):
        temp = df.drop_duplicates()
        print("After dropping duplicates:", temp.shape)
        return temp

    def restore_type(self, df):
        temp = df.copy(deep=True)
        temp['problem'] = temp['problem'].apply(lambda x: literal_eval(x))
        temp['treatment'] = temp['treatment'].apply(lambda x: literal_eval(x))
        return temp

    def drop_empty(self, df):
        temp = df[(df['problem'].apply(lambda x: len(x) > 0)) & (df['treatment'].apply(lambda x: len(x) > 0))]
        print("After dropping null:", temp.shape)
        return temp

    def apply_min_threshold(self, df):
        temp = df.copy(deep=True)

        ylb = MultiLabelBinarizer()
        y = df['treatment']
        y = ylb.fit_transform(y)
        print("Treatment labels:", len(ylb.classes_))

        y = pd.DataFrame(y)
        y.loc[:, y.sum() <= self.y_threshold] = 0
        y = y.values

        temp['treatment'] = ylb.inverse_transform(y)

        xlb = MultiLabelBinarizer()
        X = df['problem']
        X = xlb.fit_transform(X)
        print("Problem labels:", len(xlb.classes_))

        X = pd.DataFrame(X)
        X.loc[:, X.sum() <= self.X_threshold] = 0
        X = X.values

        temp['problem'] = xlb.inverse_transform(X)

        temp = self.drop_empty(temp)
        print("After applying min threshold:", temp.shape)

        ylb.fit(temp['treatment'])
        print("Treatment labels:", len(ylb.classes_))
        xlb.fit(temp['problem'])
        print("Problem labels:", len(xlb.classes_))

        temp['problem'] = temp['problem'].apply(lambda x: set(x))
        temp['treatment'] = temp['treatment'].apply(lambda x: set(x))
        temp['all_tokens'] = temp.apply(lambda x: x['problem'].union(x['treatment']),axis=1)

        return temp

    def get_preprocessed_data(self):
        df = self.load_data()
        df = self.drop_duplicates(df)
        df = self.restore_type(df)
        df = self.apply_min_threshold(df)
        return df
    

# if __name__ == "__main__":
#     preprocessor = PreprocessData(X_threshold=10, y_threshold=10)
#     preprocessor.get_preprocessed_data()