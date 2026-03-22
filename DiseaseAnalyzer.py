import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np

class DiseaseAnalyzer:
    def __init__(self, file_path):
        """
        Initializes the object with a file path and loads the data.
        """
        self.file_path = file_path
        self.df = self._read_table() # Encapsulated data loading
        self._transform_target()
        self.train_df, self.test_df = self._split_data()
        self.rf_model = None
        self._train_rf_model()

    def _read_table(self):
        """
        Internal method to read the CSV file.
        """
        return pd.read_csv(self.file_path)

    def _transform_target(self):
        """
        Binary transforms the 'target' column.
        """
        self.df['target'] = (self.df['target'] > 0).astype(int)

    def plot_pca(self):
        """
        Performs PCA and visualizes the dataset in 2D.
        """
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA execution
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualization
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k')
        plt.title(f'PCA Analysis for {self.file_path}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(*scatter.legend_elements(), title="Target")
        plt.show()

    def plot_histograms(self, layout='dodge'): # layout='dodge' or layout='stack' (appearance of the columns)
        """
        Generates histograms for all features. 
        layout: 'stack' or 'dodge' for bar positioning.
        """
        # excluding target column
        features = self.df.columns.drop('target')
    
        # having 13 sub plots
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
        axes = axes.flatten() # having iterable data structure

        for i, col in enumerate(features):
            # histogram for each trait
            # hue='target' - red for sick / blue for healthy
            
            sns.histplot(data=self.df, x=col, hue='target',multiple=f'{layout}', 
                     ax=axes[i], palette='coolwarm', bins=20)
        
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('')
            axes[i].legend(['CAD (1)', 'Healthy (0)'], title="Status")


        plt.tight_layout(pad=3.0) 
        plt.show()
    
    def _split_data(self):
        """
        Randomly splits the dataset into training (70%) and testing (30%) sets.
        """
        # Split the DataFrame into two subsets
        # test_size=0.3 ensures that 30% of the data goes to the test set
        # random_state=42 is used to ensure the split is reproducible
        train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42)
    
        return train_df, test_df
   
    def _train_rf_model(self):
        """
        Trains the random forest model
        """
        # Separate the features (X) and the labels (y) for the training set
        # The 'target' column is excluded from the features
        X_train = self.train_df.drop('target', axis=1)
        y_train = self.train_df['target']
    
        # Initialize the Random Forest Classifier
        # Using default hyperparameters and a fixed random_state for reproducibility
        self.rf_model = RandomForestClassifier(random_state=42)
    
        # Train the model on the training set
        self.rf_model.fit(X_train, y_train)

    def _predict_probabilities(self):
        """
        Predicts CAD probabilities for the test set.
        """
        # Prepare the feature matrix for the test set (exclude the label)
        X_test = self.test_df.drop('target', axis=1)
        
        # Predict the probability of having CAD (target=1) for the test set samples
        # predict_proba returns a 2D array where index 0 is prob(0) and index 1 is prob(1)
        probabilities = self.rf_model.predict_proba(X_test)[:, 1]
        return list(probabilities)
    
    

    def plot_roc(self):
        """
        Calculates and plots the Receiver Operating Characteristic (ROC) curve 
        and the Area Under the Curve (AUC).
        """
        y_probs = self._predict_probabilities()
        # Extract the true labels from the test set
        y_true = self.test_df['target']
    
        # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
        # for various threshold values
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
        # Calculate the Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)
    
        # Initialize the plot
        plt.figure(figsize=(8, 6))
    
        # Plot the ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Area Under Curve (AUC) = {roc_auc:.2f}')
    
        # Plot the diagonal dashed line (represents random guessing, AUC = 0.5)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
        # Set plot aesthetics and labels
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - specificity = False Positive Rate (FPR)')
        plt.ylabel('Sensitivity = True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
    
        plt.show()
    
        # Output the exact AUC score to the console
        print(f"The Area Under the Curve (AUC) is: {roc_auc:.4f}")
        return roc_auc
    
    

    def get_top_features(self):
        """
        Extracts and displays the top two features based on Random Forest importance scores.
        """
        # Get feature names (all columns except target)
        feature_names = self.df.drop('target', axis=1).columns
    
        # Get importance scores from the trained model
        importances = self.rf_model.feature_importances_
    
        # Create a list of (feature_name, importance_score) pairs
        feature_importance_list = list(zip(feature_names, importances))
    
        # Sort the list by importance score in descending order
        sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    
        # Get the top two
        top_two = sorted_features[:2]
    
        print("Top Two Features for this Disease Prediction:")
        for feature, score in top_two:
            print(f"- {feature}: {score:.4f}")
        
        return top_two
    


# --- Example Usage ---
#analyzer = CADAnalyzer("/Users/iddocohen/Documents/2025:6 A/laboratory for bioinformatics tools/ex5/CAD.csv")
#analyzer.plot_pca()
#analyzer.plot_histograms(layout='dodge')
#analyzer.plot_roc()
#analyzer.get_top_features()
