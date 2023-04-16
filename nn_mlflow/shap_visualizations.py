from matplotlib import pyplot as plt
import numpy as np
import shap

PLOTS_PATH = "plots/"

def create_force_plot(model, x, output_file):
    
    # Create a TreeExplainer object using the SHAP library
    explainer = shap.TreeExplainer(model)
        
    # Compute the SHAP values for the chosen data point
    shap_values = explainer.shap_values(x)
        
    # Visualize the explanations for the chosen data point
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        x,
        matplotlib=True,
        show=False
    )
    
    # Save the plot to a file
    plt.savefig(PLOTS_PATH + output_file)
    
def create_summary_plot(model, X, output_file):

    # Create a TreeExplainer object using the SHAP library
    explainer = shap.TreeExplainer(model)
        
    # Compute the SHAP values for all data points
    shap_values = explainer.shap_values(X)
        
    # Visualize the explanations for all data points
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )
    
    # Save the plot to a file
    plt.savefig(PLOTS_PATH + output_file)
    
def create_class_summary_plot(model, X, y, output_file):
    
    # Create a TreeExplainer object using the SHAP library
    explainer = shap.TreeExplainer(model)
        
    # Compute the SHAP values for all data points
    shap_values = explainer.shap_values(X)
    
    # Get the class names from the target variable
    class_names = np.unique(y)

    # Create a summary plot for each class
    for class_name in class_names:
        class_indices = np.where(y == class_name)[0]
        shap.summary_plot(
            shap_values[class_indices],
            X[class_indices],
            show=False
        )
        
        # Save the plot to a file
        plt.savefig(PLOTS_PATH + output_file)
