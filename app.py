import streamlit as st
import pandas as pd
import pickle
import re

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="centered"
)

# Load the saved models and vectorizer
@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open('chronic_disease_model.pkl', 'rb') as f:
        chronic_model = pickle.load(f)
    
    with open('contagious_disease_model.pkl', 'rb') as f:
        contagious_model = pickle.load(f)
    
    # Load the dataset for symptom matching
    df = pd.read_csv('Diseases_Symptoms.csv')
    return tfidf_vectorizer, chronic_model, contagious_model, df

# Extract all unique symptoms from the dataset
@st.cache_data
def extract_symptoms(df):
    all_symptoms = []
    for symptom_text in df['Symptoms']:
        # Split by commas and clean each symptom
        symptoms = [s.strip() for s in symptom_text.split(',')]
        all_symptoms.extend(symptoms)
    
    # Remove duplicates and sort
    unique_symptoms = sorted(list(set(all_symptoms)))
    return unique_symptoms

# Simple text cleaning function (no NLTK dependency)
def simple_clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to predict disease based on symptoms
def predict_disease(symptoms_text, tfidf_vectorizer, df, top_n=5):
    # Clean and vectorize the input symptoms
    cleaned_input = simple_clean_text(symptoms_text)
    input_vector = tfidf_vectorizer.transform([cleaned_input])
    
    # Calculate similarity with all diseases
    similarity_scores = []
    
    for idx, row in df.iterrows():
        # Get cleaned symptoms for this disease
        disease_symptoms = row['Symptoms']
        cleaned_symptoms = simple_clean_text(disease_symptoms)
        disease_vector = tfidf_vectorizer.transform([cleaned_symptoms])
        
        # Calculate cosine similarity
        similarity = (input_vector * disease_vector.T).toarray()[0][0]
        similarity_scores.append((row['Name'], similarity, row['Treatments'], 
                                 row['Contagious'], row['Chronic']))
    
    # Sort by similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N matches
    return similarity_scores[:top_n]

# Main function
def main():
    # Load models and data
    try:
        tfidf_vectorizer, chronic_model, contagious_model, df = load_models()
        models_loaded = True
        
        # Extract unique symptoms
        unique_symptoms = extract_symptoms(df)
        
        # Create common symptoms list (most frequent symptoms)
        common_symptoms = [
            "fever", "cough", "headache", "fatigue", "nausea", "vomiting", 
            "chest pain", "shortness of breath", "abdominal pain", "dizziness",
            "rash", "joint pain", "muscle pain", "sore throat", "runny nose"
        ]
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models_loaded = False
    
    # App title and description
    st.title("🏥 Disease Prediction System")
    st.write("""
    This application helps identify possible diseases based on symptoms. 
    Enter your symptoms below to get predictions.
    """)
    
    # Disclaimer
    st.warning("""
    **Medical Disclaimer**: This tool is for educational purposes only and is not a substitute 
    for professional medical advice, diagnosis, or treatment. Always seek the advice of your 
    physician or other qualified health provider with any questions you may have regarding a 
    medical condition.
    """)
    
    if models_loaded:
        # Create two tabs for different input methods
        tab1, tab2 = st.tabs(["Text Input", "Symptom Selection"])
        
        with tab1:
            # Original text input method
            symptoms_input_text = st.text_area(
                "Enter your symptoms (separated by commas):", 
                "fever, cough, fatigue", 
                height=100
            )
            
            # Show symptom suggestions
            with st.expander("Need help with symptoms? Click here for suggestions"):
                st.write("Common symptoms you can include:")
                st.write(", ".join(common_symptoms))
                
                # Allow searching for specific symptoms
                symptom_search = st.text_input("Search for specific symptoms:")
                if symptom_search:
                    filtered_symptoms = [s for s in unique_symptoms if symptom_search.lower() in s.lower()]
                    if filtered_symptoms:
                        st.write("Matching symptoms:")
                        st.write(", ".join(filtered_symptoms[:20])) # Limit to 20 results
                    else:
                        st.write("No matching symptoms found.")
            
            # Prediction button for text input
            if st.button("Predict Diseases (Text Input)"):
                if symptoms_input_text.strip():
                    with st.spinner("Analyzing symptoms..."):
                        # Get predictions
                        predictions = predict_disease(symptoms_input_text, tfidf_vectorizer, df)
                        display_predictions(predictions)
                else:
                    st.error("Please enter some symptoms.")
        
        with tab2:
            # Multi-select widget for symptoms
            selected_symptoms = st.multiselect(
                "Select your symptoms:",
                options=unique_symptoms,
                default=["fever", "cough"],
                help="Select multiple symptoms from the dropdown"
            )
            
            # Quick selection of common symptoms
            #st.write("Quick add common symptoms:")
            #cols = st.columns(3)
            #for i, symptom in enumerate(common_symptoms):
             #   if i % 3 == 0:
              #      with cols[0]:
               #         if st.button(symptom, key=f"btn_{symptom}"):
                #            if symptom not in selected_symptoms:
                 #               selected_symptoms.append(symptom)
                  #              st.experimental_rerun()
                #elif i % 3 == 1:
                 #   with cols[1]:
                  #      if st.button(symptom, key=f"btn_{symptom}"):
                   #         if symptom not in selected_symptoms:
                    #            selected_symptoms.append(symptom)
                     #           st.experimental_rerun()
                #else:
                 #   with cols[2]:
                  #      if st.button(symptom, key=f"btn_{symptom}"):
                   #         if symptom not in selected_symptoms:
                    #            selected_symptoms.append(symptom)
                     #           st.experimental_rerun()
            
            # Display selected symptoms
            if selected_symptoms:
                st.write("Your selected symptoms:")
                st.write(", ".join(selected_symptoms))
                
                # Prediction button for selected symptoms
                if st.button("Predict Diseases (Selected Symptoms)"):
                    with st.spinner("Analyzing symptoms..."):
                        # Convert selected symptoms to comma-separated string
                        symptoms_text = ", ".join(selected_symptoms)
                        # Get predictions
                        predictions = predict_disease(symptoms_text, tfidf_vectorizer, df)
                        display_predictions(predictions)
            else:
                st.info("Please select at least one symptom.")

# Function to display predictions (extracted to avoid code duplication)
def display_predictions(predictions):
    # Display results
    st.subheader("Possible Diseases:")
    
    # Create tabs for each disease
    tabs = st.tabs([f"{disease}" for disease, _, _, _, _ in predictions])
    
    for i, (disease, score, treatment, contagious, chronic) in enumerate(predictions):
        with tabs[i]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Match Score**: {score:.2f}")
                st.markdown(f"**Contagious**: {'Yes' if contagious == 'Yes' else 'No'}")
                st.markdown(f"**Chronic**: {'Yes' if chronic == 'Yes' else 'No'}")
            
            with col2:
                st.markdown("### Treatments")
                st.write(treatment)

# Run the app
if __name__ == "__main__":
    main()