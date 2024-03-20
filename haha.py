from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix, hstack

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Define preprocessing function
def preprocess_text(X_train):
    # Add your preprocessing steps here (e.g., tokenization, stopword removal, etc.)
    # X_train_tfidf = tfidf_vectorizer.transform(X_train)
    return X_train

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    request_data = request.json
    var=request_data['name']
    
    
    # Print the entire JSON data received from the request
    # Preprocess the input text
    
    # Vectorize the preprocessed text using TF-IDF vectorizer
    var=preprocess_text([var])

    row_indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    column_indices = [1, 0, 913, 875, 592, 834, 610, 286, 529, 994, 766, 229, 609, 749, 58]
    data_values = [0.05310369965886899, 0.43544770531666566, 0.26498914839037896, 0.22007164576396662, 0.24599927641720076, 0.1969237906959751, 0.2559899910547987, 0.1746110124479404, 0.24151935235575064, 0.2706544520394489, 0.27776339863855015, 0.2706544520394489, 0.40261316596351326, 0.2060471767580278, 0.08399667847383807]

    # Convert the lists into numpy arrays
    row_indices = np.array(row_indices)
    column_indices = np.array(column_indices)
    data_values = np.array(data_values)

    # Create the CSR matrix
    csr_matrix_obj = csr_matrix((data_values, (row_indices, column_indices)))
    print(var)

    num_dummy_features = 1000 - csr_matrix_obj.shape[1]
    dummy_features = csr_matrix((csr_matrix_obj.shape[0], num_dummy_features))

    # Horizontally stack the input matrix with the dummy features
    X_train_tfidf_with_dummy = hstack([csr_matrix_obj, dummy_features])
        
    # Make predictions using the loaded model
    prediction = loaded_model.predict(X_train_tfidf_with_dummy)
    print(prediction)

    prediction = prediction.tolist()

    
    # Convert the prediction to a human-readable label
    # (You may need to adapt this depending on your specific problem)
    


    
    # Return the prediction as JSON response
    response = {'prediction': prediction}

    # Return the JSON response using jsonify
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
