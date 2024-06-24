from flask import Flask, render_template
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

app = Flask(__name__)

# Dataframe contoh
data = pd.DataFrame({
    'TROMBOSIT': np.random.rand(100),
    'HCT': np.random.rand(100),
    'IgG': np.random.rand(100),
    'IgM': np.random.rand(100),
    'DEMAM': np.random.randint(0, 2, 100),
    'JENIS KELAMIN': np.random.randint(0, 2, 100),
    'UMUR': np.random.randint(1, 100, 100),
    'LABEL': np.random.randint(0, 2, 100)
})

# Memisahkan fitur dan label
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Membuat confusion matrix dan laporan klasifikasi
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# # Menyimpan confusion matrix sebagai gambar
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.savefig('static/images/confusion_matrix.png')

@app.route('/')
def index():
    return render_template('/index.html', class_report=class_report)

if __name__ == '__main__':
    app.run(debug=True)

