import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import tkinter as tk
from tkinter import ttk

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
x = np.array(data.drop([predict], axis=1))

y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y , test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

accuracy =  linear.score(x_test, y_test)
print(accuracy)

print("Coefficient: \n", linear.coef_)
print("intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
# Create a DataFrame for predictions and actual values
results = pd.DataFrame(x_test, columns=["G1", "G2", "Study Time", "Failures", "Absences"])
results["Predicted G3"] = predictions
results["Actual G3"] = y_test

# Print the results as a table
print("\nPredictions Table:")
print(results.to_string(index=False))

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Create a DataFrame for predictions and actual values
results = pd.DataFrame(x_test, columns=["G1", "G2", "Study Time", "Failures", "Absences"])
results["Predicted G3"] = predictions
results["Actual G3"] = y_test

# Function to display the table in a pop-out UI
def show_table():
    # Create a new window
    root = tk.Tk()
    root.title("Predictions Table")

    # Create a Treeview widget
    tree = ttk.Treeview(root, columns=list(results.columns), show='headings')
    tree.pack(fill=tk.BOTH, expand=True)

    # Add column headings
    for col in results.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Insert rows into the table
    for _, row in results.iterrows():
        tree.insert("", tk.END, values=row.tolist())

    # Run the application
    root.mainloop()

# Call the function to show the table
show_table()

###############################################
# # 3D plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # Scatter plot with G1, G2, and G3
# ax.scatter(data['G1'], data['G2'], data['G3'], c=data['G3'], cmap='viridis', s=50)
#
# ax.set_xlabel("G1 (First Grade)")
# ax.set_ylabel("G2 (Second Grade)")
# ax.set_zlabel("G3 (Final Grade)")
# plt.title("3D Scatter Plot: G1, G2, G3")
# plt.show()

