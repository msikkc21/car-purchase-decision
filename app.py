import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

# Membaca dataset
data = pd.read_csv('./car_data.csv')

# Menghapus baris dengan nilai yang hilang
data.dropna(inplace=True)

# Menghapus baris duplikat
data.drop_duplicates(inplace=True)

# Menghapus outlier berdasarkan IQR untuk kolom 'Age'
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Age'] >= (Q1 - 1.5 * IQR)) & (data['Age'] <= (Q3 + 1.5 * IQR))]

# Menghapus outlier berdasarkan IQR untuk kolom 'AnnualSalary'
Q1 = data['AnnualSalary'].quantile(0.25)
Q3 = data['AnnualSalary'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['AnnualSalary'] >= (Q1 - 1.5 * IQR)) & (data['AnnualSalary'] <= (Q3 + 1.5 * IQR))]

# Mengubah nilai kategori menjadi numerik
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Menampilkan data yang telah dibersihkan
print(data.head())
X = data[['Gender', 'Age', 'AnnualSalary']]  # Fitur
y = data['Purchased']             # Target


# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Melatih model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Evaluasi model
y_pred = rf_model.predict(X_test)

cm = [[0,0],[0,0]]
for true, pred in zip(y_test, y_pred):
    if true == 0 and pred == 0: # True Negatif
        cm[0][0] += 1
    elif true == 0 and pred == 1:  # False Positif
        cm[0][1] += 1
    elif true == 1 and pred == 0:  # False Negatif
        cm[1][0] += 1
    elif true == 1 and pred == 1:  # True Positif
        cm[1][1] += 1

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

accuracy = accuracy * 100
recall = recall * 100
specificity = specificity * 100
precision = precision * 100
f1_score = f1_score * 100


# Cetak Hasil
print("Hasil Evaluasi Model Random Forest")
print("Confusion Matrix:")
print(f"[[{tn} {fp}]")
print(f" [{fn} {tp}]]")
print(f"Accuracy    : {accuracy:.2f}%")
print(f"Recall      : {recall:.2f}%")
print(f"Specificity : {specificity:.2f}%")
print(f"Precision   : {precision:.2f}%")
print(f"F1-Score    : {f1_score:.2f}%")


# FLASK START
app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        gender = request.form.get("gender")
        age = request.form.get("age")
        salary = request.form.get("salary")

        if gender == "Male" :
            gender = 1
        elif gender =="Female" :
            gender = 0
        age = float(age)
        salary = float(salary)

        new_data = pd.DataFrame([[gender, age, salary]], columns=['Gender', 'Age', 'AnnualSalary'])  # Data baru

        prediction_rf = rf_model.predict(new_data)
        prediction_proba_rf = rf_model.predict_proba(new_data)

        # Menampilkan hasil prediksi
        prediksi = "Purchased" if prediction_rf[0] == 1 else "Not Purchased"
        prob_beli = prediction_proba_rf[0][1] * 100
        prob_tidak = prediction_proba_rf[0][0] * 100

        return render_template("index.html", prediction = prediksi, accuracy = accuracy, beli = prob_beli, tidak = prob_tidak, f1 = f'{f1_score:.2f}', recall = f'{recall:.2f}', specificity = specificity, precision = f'{precision:.2f}')
    return render_template("index.html", prediction = None)

if __name__ == "__main__":
    app.run( host= "127.0.0.9", port= 8080, debug=True)