import pandas
from sklearn.preprocessing import StandardScaler


def preprocess_metadata(df):
    # 1. Binary encoding
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    df['ParalysisSide'] = df['ParalysisSide'].map({'left': 0, 'right': 1})
    df['Handedness'] = df['Handedness'].map({'left': 0, 'right': 1})
    df['IsFirstTime'] = df['IsFirstTime'].map({'no': 0, 'yes': 1})

    # 2. Simple One-Hot for StrokeLocation (Key locations)
    locations = ['pons', 'basal ganglia', 'thalamus', 'ventricular', 'cerebellum']
    for loc in locations:
        df[f'Loc_{loc}'] = df['StrokeLocation'].str.lower().apply(lambda x: 1 if loc in x else 0)

    # 3. Scaling numerical features
    num_cols = ['Age', 'Duration', 'NIHSS', 'MBI', 'mRS']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Drop unused columns
    drop_cols = ['Participant_ID', 'StrokeLocation']
    df_final = df.drop(columns=drop_cols)
    return df_final

if __name__ == '__main__':
    df = pandas.read_csv('../data/participants.csv')
    df = preprocess_metadata(df)
    df.to_csv('preprocessed.csv', index=False)