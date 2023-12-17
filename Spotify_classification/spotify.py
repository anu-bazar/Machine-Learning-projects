
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from statistics import median, mode

data = pd.read_csv('/content/dataset.csv')
data.head(20)


# Visualize the distribution of songs in each genre
plt.figure(figsize=(12, 6))
sns.countplot(x='track_genre', data=data, order=data['track_genre'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.title('Number of Songs in Each Genre')
plt.show()


# Visualize the distribution of popularity scores
plt.figure(figsize=(8, 6))
sns.histplot(data['popularity'], bins=20, kde=True)
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Popularity Scores')
plt.show()


# Visualize the distribution of danceability scores
plt.figure(figsize=(8, 6))
sns.histplot(data['danceability'], bins=20, kde=True)
plt.xlabel('Danceability Score')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability Scores')
plt.show()


# Visualize the distribution of energy scores
plt.figure(figsize=(8, 6))
sns.histplot(data['energy'], bins=20, kde=True)
plt.xlabel('Energy Score')
plt.ylabel('Frequency')
plt.title('Distribution of Energy Scores')
plt.show()



# Visualize the distribution of valence scores
plt.figure(figsize=(8, 6))
sns.histplot(data['valence'], bins=20, kde=True)
plt.xlabel('Valence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Valence Scores')
plt.show()



# Convert duration from milliseconds to minutes
data['duration_minutes'] = data['duration_ms'] / 60000  # 1 minute = 60,000 milliseconds
# Plot
plt.figure(figsize=(8, 6))
sns.histplot(data['duration_minutes'], bins=100, kde=True)
plt.xlabel('Track Length (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Track Lengths (in Minutes)')
plt.show()


# Convert duration from milliseconds to minutes
data['duration_minutes'] = data['duration_ms'] / 60000  # 1 minute = 60,000 milliseconds
plt.figure(figsize=(8, 6))
sns.histplot(data['duration_minutes'], bins=100, kde=True)
plt.xlabel('Track Length (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Track Lengths (in Minutes)')
plt.xlim(0, 10)
plt.show()


# Stats 101
data['duration_minutes'] = data['duration_ms'] / 60000
average_duration_minutes = data['duration_minutes'].mean()
median_duration_minutes = median(data['duration_minutes'])
mode_duration_minutes = mode(data['duration_minutes'])
print(f'The average value is {average_duration_minutes}, median value is {median_duration_minutes}, and the mode is {mode_duration_minutes}.')


#num of songs/explicit
plt.figure(figsize=(8, 6))
sns.countplot(x='explicit', data=data)
plt.xlabel('Explicit')
plt.ylabel('Number of Songs')
plt.title('Explicit vs. Non-Explicit Songs')
plt.show()


#tempo
plt.figure(figsize=(8, 6))
sns.histplot(data['tempo'], bins=20, kde=True)
plt.xlabel('Tempo (BPM)')
plt.ylabel('Frequency')
plt.title('Distribution of Tempo')
plt.show()

#tempo by genre: Techno
query_data = data[data['track_genre'] == 'jazz']
average_techno_tempo = query_data['tempo'].mean()
print("Average Tempo for Techno Genre:", average_techno_tempo)


#tempo for every genre
average_tempo_by_genre = data.groupby('track_genre')['tempo'].mean()
print(average_tempo_by_genre)


#time signature
time_signature_mapping = {
    1: '1/4',
    3: '3/4',
    4: '4/4',
    5: '5/4',
    7: '7/4'
}

data['time_signature'] = data['time_signature'].map(time_signature_mapping)
plt.figure(figsize=(10, 6))
sns.countplot(x='time_signature', data=data, order=data['time_signature'].value_counts().index)
plt.xlabel('Time Signature')
plt.ylabel('Number of Songs')
plt.title('Distribution of Time Signatures')
plt.xticks(rotation=45)
plt.show()

# primary artist
def extract_primary_artist(artist_string):
    if pd.isna(artist_string):
        return artist_string
    artists = artist_string.split(';')
    return artists[0].strip()

data['primary_artist'] = data['artists'].apply(extract_primary_artist)
print(data.head())


# explicit column from boolean to 0/1s
data['explicit'] = data['explicit'].fillna(False).astype(int) #includes NaN
data.head()

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'explicit', 'liveness', 'valence', 'time_signature']

fig = make_subplots(cols=4, rows=3, subplot_titles=features)
for i, feature in enumerate(features):
    fig.add_trace(go.Histogram(x=data[feature], name=feature), row=i//4 + 1, col=i%4 + 1)

width = 1000
height = 800

fig.update_layout(width=width, height=height, title='Feature Distribution')


import plotly.graph_objs as go
from plotly.subplots import make_subplots

features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'explicit', 'liveness', 'valence', 'time_signature']

fig = make_subplots(cols=4, rows=3, subplot_titles=features)

for i, feature in enumerate(features):
    fig.add_trace(go.Histogram(x=data[data['track_genre'] == 'techno'][feature], name=feature), row=i//4 + 1, col=i%4 + 1)

width = 1000
height = 800

fig.update_layout(width=width, height=height, title='Feature Distribution in Techno Genre')
fig.show()

import plotly.express as px
correlation_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
            'loudness', 'valence', 'tempo', 'time_signature']
fig = px.imshow(data[correlation_features].corr())
fig.update_layout(title='Feature Correlation Heatmap')
fig.show()

# drop loudness
data.drop('loudness', axis=1, inplace=True)
data.head()

import plotly.express as px
correlation_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
             'valence', 'tempo', 'time_signature']
fig = px.imshow(data[correlation_features].corr())
fig.update_layout(title='Feature Correlation Heatmap')
fig.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data = pd.read_csv('dataset.csv')
selected_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'valence', 'tempo']
X = data[selected_features]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

n_components = 2  # 2-D
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(X_std)
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(n_components)])
colors = np.where(pca_df['PC1'] >= pca_df['PC2'], 'r', 'b')
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=colors, alpha=0.5)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('PCA Visualization of Spotify Data (PC1 vs. PC2)')
plt.show()

loadings = pca.components_
loadings_df = pd.DataFrame(data=loadings, columns=selected_features, index=[f'PC{i + 1}' for i in range(n_components)])
print("PC1:")
print(loadings_df.loc['PC1'])
print("PC2")
print(loadings_df.loc['PC2'])

# Song title
song_title = "I'm Yours"  # CHANGE HERE!!

# search
matching_songs = data[data['track_name'].str.contains(song_title, case=False, na=False)]
top_10_matching_songs = matching_songs[['track_name', 'artists', 'track_id']][:10]
print("Top 10 matching songs:")
print(top_10_matching_songs)

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
selected_features = ['tempo', 'danceability', 'energy', 'valence', 'duration_ms', 'popularity', 'speechiness', 'liveness', 'acousticness', 'instrumentalness']
# track ID
reference_track_id = "3S0OXQeoh0w6AY8WQVckRW"  # CHANGE HERE!! - Jason Mraz I'm yours
reference_song = data[data['track_id'] == reference_track_id]
scaler = StandardScaler()
X = scaler.fit_transform(data[selected_features])
# cosine similarity
cosine_sim = cosine_similarity([X[reference_song.index[0]]], X)
similarity_scores = pd.DataFrame(data={'Track Name': data['track_name'], 'Artists': data['artists'], 'Similarity Score': cosine_sim[0]})
similar_songs = similarity_scores.sort_values(by='Similarity Score', ascending=False)
N = 10 # number of songs
top_n_recommendations = similar_songs.head(N)
print(f"We found {N} songs similar to this:")
for index, row in top_n_recommendations.iterrows():
    print(f"{row['Artists']} - {row['Track Name']}")

X = data.iloc[:5000].drop(['track_genre'], axis=1)  # features
y = data.iloc[:5000]['track_genre']  # target
#randomize
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report)
print('Confusion Matrix:\n', confusion_mat)

subset_data = data.iloc[:5000]

X = subset_data.drop(['track_genre'], axis=1)  # features
y = subset_data['track_genre']  # target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_gb = gb_classifier.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
classification_report_gb = classification_report(y_test, y_pred_gb)
confusion_mat_gb = confusion_matrix(y_test, y_pred_gb)

print('Gradient Boosting Classifier Results:')
print(f'Accuracy: {accuracy_gb}')
print('Classification Report:\n', classification_report_gb)
print('Confusion Matrix:\n', confusion_mat_gb)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

subset_data = data.iloc[:5000]

X = subset_data.drop(['track_genre'], axis=1)  # features
y = subset_data['track_genre']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

numeric_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(len(numeric_features),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


y_train_onehot = keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test_onehot = keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))
model.fit(X_train, y_train_onehot, epochs=20, batch_size=32, validation_data=(X_test, y_test_onehot))

loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f'Accuracy: {accuracy}')

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
classification_report = classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_)
confusion_mat = confusion_matrix(y_test, y_pred_labels)
print('Classification Report:\n', classification_report)
print('Confusion Matrix:\n', confusion_mat)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
subset_data = data.iloc[:5000]

X = subset_data.drop(['track_genre'], axis=1)  # features
y = subset_data['track_genre']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print('MLP Classifier Results:')
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report)
print('Confusion Matrix:\n', confusion_mat)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
numeric_features = ['popularity', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
X = data[numeric_features]
X_train, X_test, y_train, y_test = train_test_split(X.drop('popularity', axis=1), X['popularity'], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

numeric_features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
X = data[numeric_features]
y = data['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

numeric_features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
X = data[numeric_features]
y = data['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', confusion_mat)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
numeric_features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
X = data[numeric_features]
y = data['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')


import pandas as pd
genre_avg = data.groupby('track_genre')[['danceability', 'energy', 'valence']].mean()
print(genre_avg)

import pandas as pd
top_danceability = genre_avg.sort_values(by='danceability', ascending=False).head(5)
top_energy = genre_avg.sort_values(by='energy', ascending=False).head(5)
top_valence = genre_avg.sort_values(by='valence', ascending=False).head(5)
print("Top 5 Genres by Danceability:")
print(top_danceability)
print("\nTop 5 Genres by Energy:")
print(top_energy)
print("\nTop 5 Genres by Valence:")
print(top_valence)

most_energetic_genres = genre_avg.sort_values(by='energy', ascending=False)
most_danceable_genres = genre_avg.sort_values(by='danceability', ascending=False)
highest_valence_genres = genre_avg.sort_values(by='valence', ascending=False)
print("Most Energetic Genres:")
print(most_energetic_genres)
print("\nMost Danceable Genres:")
print(most_danceable_genres)
print("\nGenres with the Highest Valence:")
print(highest_valence_genres)
