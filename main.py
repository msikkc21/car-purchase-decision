import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
data = pd.read_csv('./car_data.csv')

# Menghapus baris dengan nilai yang hilang
data.dropna(inplace=True)

# Menghapus baris duplikat
data.drop_duplicates(inplace=True)

# Mengubah kolom 'Gender' menjadi kategori
data['Gender'] = data['Gender'].astype('category')

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
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Menampilkan hasil evaluasi
print("Akurasi Model: {:.2f}%".format(accuracy * 100))
print("F1-Score: {:.2f}".format(f1))
print("Recall: {:.2f}".format(recall))
print("\nClassification Report:\n", report)
    
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Purchased', 'Not Purchased'], yticklabels=['Purchased', 'Not Purchased'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

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
        else :
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

        accuracy = accuracy_score(y_test, y_pred) * 100
        return render_template("index.html", prediction = prediksi, accuracy = accuracy, beli = prob_beli, tidak = prob_tidak, f1 = f1, recall = recall)
    return render_template("index.html", prediction = None)

if __name__ == "__main__":
    app.run( host= "127.0.0.9", port= 8080, debug=False)