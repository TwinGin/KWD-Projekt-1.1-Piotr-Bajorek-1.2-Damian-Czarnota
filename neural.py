import pandas as pd

dataset = pd.read_csv("bank-full.csv", sep=";")
data = dataset.iloc[:, :8].values
verification = dataset.iloc[:, 16].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in range(1, 8):
    encoderData = LabelEncoder()
    data[:, i] = encoderData.fit_transform(data[:, i])

encoderData2 = OneHotEncoder(categorical_features=[1])
data = encoderData2.fit_transform(data).toarray()
data = data[:, 0:]
encoderVerification = LabelEncoder()
verification[:] = encoderVerification.fit_transform(verification[:])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, verification, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
neural = Sequential([
    Dense(16, input_shape=(19,)),
    Activation('relu'),
    #Dropout(0.1),
    Dense(15),
    Activation('softmax'),
    #Dropout(0.05),
    Dense(1),
    Activation('sigmoid'),
])
neural.summary()
neural.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
run_hist = neural.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test))

score = neural.evaluate(X_train, Y_train)
print("Dokladnosc:")
print(score[1])
Y_pred = neural.predict(X_test)
Y_pred = (Y_pred > 0.5)
Y_test = (Y_test > 0.5)
import matplotlib.pyplot as plt
plt.plot(run_hist.history["loss"], 'r', marker='.', label="Train loss")
plt.plot(run_hist.history["val_loss"], 'b', marker='.', label="Validation loss")
plt.title("Train loss and validation error")
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('Error')
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Macierz")
print(cm)

