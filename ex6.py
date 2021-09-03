#%%
# recognition of blurred images of digits from 0 to 9
import matplotlib.pyplot as  plt
from numpy.lib.npyio import load 
from sklearn.datasets import load_digits
digits = load_digits()
#dir(digits)
#plt.gray()
#plt.matshow(digits.images[1])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
 # len(x_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
model.predict(digits.data[0:5])
y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# %%
