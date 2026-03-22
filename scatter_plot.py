import pandas as pd
# input: path of a csv file representing samples of people
# output: df representing the file
def read_table(path):
	return pd.read_csv(path)

# input: df representing a file - samples of people
# output: df representing the file - the target col == 1 iff the patient is sick
def transform_target(df):
	df_copy = df.copy()
	df_copy.loc[df_copy['target'] > 0, 'target'] = 1
	return df_copy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# input: df representing the file - the target col == 1 iff the patient is sick
# output: printing a scatter plot of 2 dimensions (PCA with 2 dimensions onl)
def perform_pca_and_plot(df):
    # seperating cols to x and y(sick/ health)
    X = df.drop('target', axis=1) 
    y = df['target']
    
    # standardization to x
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # having a PCA with 2 dimensions only
    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X_scaled)
    
    # Scatter Plot
    plt.figure(figsize=(10, 7))
    # setting the sick points (target == 1) as red
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
    
    plt.title('2D PCA of Heart Disease Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(*scatter.legend_elements(), title="Target (0=Healthy, 1=CAD)")
    plt.grid(True)
    plt.show()
    
    return X_pca



# having a scatter plot of the CAD.csv file
x = perform_pca_and_plot(transform_target(read_table("/Users/iddocohen/Documents/2025:6 A/laboratory for bioinformatics tools/ex5/CAD.csv")))
