import torch
from torch_geometric.utils import from_smiles
import streamlit as st

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('resources/model_lipo_tuned.pth', map_location=device)
model.eval()

# Function to predict lipophilicity
def predict_lipophilicity(smiles_list):
    predictions = []
    for smile in smiles_list:
        try:
            # Convert SMILES to graph
            g = from_smiles(smile)
            g.x = g.x.float()
            g = g.to(device)
            
            # Add batch information
            g.batch = torch.tensor([0] * g.num_nodes, dtype=torch.long, device=device)

            # Perform prediction
            with torch.no_grad():
                pred = model(g.x, g.edge_index, g.edge_attr, g.batch)
            predictions.append(pred.item())
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
    return predictions

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    # Streamlit UI for prediction
    st.title('Lipophilicity Prediction from SMILES')
    st.write('Enter SMILES strings below and click "Predict" to get the lipophilicity values.')

    smiles_input = st.text_area("SMILES Strings (one per line)")
    if st.button("Predict"):
        if smiles_input:
            smiles_list = smiles_input.split('\n')
            lipophilicity_predictions = predict_lipophilicity(smiles_list)
            
            st.write("### Predictions")
            for smile, pred in zip(smiles_list, lipophilicity_predictions):
                if isinstance(pred, float):
                    st.write(f"**SMILES:** {smile} \n **Predicted Lipophilicity:** {pred:.4f}")
                else:
                    st.write(f"**SMILES:** {smile} \n **Error:** {pred}")
        else:
            st.write("Please enter at least one SMILES string.")

elif page == "About":
    # About page content
    st.title("About This App")
    st.write("""
    This web application predicts the lipophilicity of molecules from their SMILES representations using a Graph Neural Network (GNN) model.

    ### How It Works
    1. **SMILES Input**: Users can input one or multiple SMILES strings (one per line) in the text area.
    2. **Prediction**: The app converts these SMILES strings into graph representations and feeds them into a pre-trained GNN model.
    3. **Output**: The model predicts the lipophilicity values for the given SMILES strings.

    ### About Lipophilicity
    Lipophilicity is a measure of how well a substance can dissolve in fats, oils, lipids, and non-polar solvents such as hexane or toluene. It is a critical property in drug design, affecting a drug's absorption, distribution, metabolism, and excretion (ADME) properties.

    ### About the Model
    The GNN model used in this app is trained on a dataset of molecules with known lipophilicity values. It leverages the power of neural networks to learn complex relationships between the molecular structure and its lipophilicity.

    ### About Us
    This app is developed as a part of a project to demonstrate the application of machine learning in cheminformatics. For more information, visit our [GitHub repository](https://github.com/codetodiscovery/Lipophilicity-GNN.git).

    ### Contact
    For any queries or feedback, please contact us at [codetodiscovery@gmail.com](mailto:your-email@example.com).
    """)

