import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import warnings

warnings.filterwarnings('ignore') # Suppress warnings, e.g., from TextBlob

# Define stopwords once
stop_words = set(stopwords.words('english'))

# --- Helper Functions (for this module's internal use) ---

# TextBlob Sentiment (used for comparison)
def get_textblob_sentiment_local(text):
    if not isinstance(text, str):
        text = str(text)
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.05:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Text Cleaning for NLP (Word Clouds, etc.)
def clean_text_local(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Function to generate word cloud (local to this module)
def generate_wordcloud_local(text_data, title="Word Cloud"):
    if not text_data:
        st.warning(f"No text data available for {title}.")
        return

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        collocations=False,
        colormap='magma'
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=10)
    st.pyplot(fig)
    plt.close(fig)

# --- Main Dashboard Function for ML Analysis ---
def show_ml_analysis_dashboard(df_full, classification_model, regression_model, regression_scaler):
    st.title("🤖 ML & NLP Insights: Predictive Power & Deep Text Analysis")
    st.info("""
    This section showcases our advanced Machine Learning models for predicting product ratings and classifying review sentiment,
    along with deeper Natural Language Processing insights.
    """)

    # --- Sample Size Selector for ML/NLP Section ONLY ---
    st.markdown("---")
    with st.expander("⚙️ **ML/NLP Data Settings (for performance)**"):
        st.write("Adjust the number of reviews used for ML model predictions and text analysis in this section. Higher sample sizes mean more accurate insights but slower processing.")
        
        # Max value for slider should be the actual size of the full DataFrame passed
        max_rows_for_ml_sampling = len(df_full)
        
        ml_sample_size = st.slider(
            "Select ML/NLP Data Sample Size:",
            min_value=100, # Minimum sensible sample
            max_value=max_rows_for_ml_sampling,
            value=min(10000, max_rows_for_ml_sampling), # Default to 10k or max available if less
            step=100,
            format='%d rows',
            key='ml_sample_size_slider'
        )
        
        # Apply sampling based on the slider value
        if len(df_full) > ml_sample_size:
            df = df_full.sample(n=ml_sample_size, random_state=42).reset_index(drop=True)
            st.info(f"Using a sample of {ml_sample_size:,} reviews for ML/NLP analysis.")
        else:
            df = df_full.copy() # Use full data if smaller than sample size
            st.info(f"Using all {len(df):,} reviews for ML/NLP analysis.")

    st.markdown("---")

    # --- Row 1: Sentiment Classification Model & Evaluation ---
    st.header("1. Sentiment Classification Model")
    st.write("Leveraging `classification_pipeline.pkl` to categorize reviews as Positive, Neutral, or Negative.")

    col_clf1, col_clf2 = st.columns(2)

    with col_clf1:
        st.subheader("Model-Predicted Sentiment Distribution")
        if not df.empty:
            with st.spinner("Predicting sentiment with classification model..."):
                df['Predicted_Sentiment_ML'] = classification_model.predict(df['Text'].astype(str).tolist())

            predicted_sentiment_counts = df['Predicted_Sentiment_ML'].value_counts().reset_index()
            predicted_sentiment_counts.columns = ['Sentiment', 'Count']
            fig_pred = px.bar(predicted_sentiment_counts, x='Sentiment', y='Count',
                            color='Sentiment', title='Model Predicted Sentiment Distribution',
                            color_discrete_map={'Positive': 'darkgreen', 'Negative': 'darkred', 'Neutral': 'darkblue'},
                            height=400)
            st.plotly_chart(fig_pred, use_container_width=True)
            st.info("💡 **Insight:** This distribution gives a real-time pulse of customer sentiment as interpreted by our advanced model.")
        else:
            st.warning("No data available for ML classification. Adjust sample size if needed.")

    with col_clf2:
        st.subheader("Model Performance Evaluation")
        st.write("Evaluating the classification model's accuracy, precision, recall, and F1-score.")
        if not df.empty and 'Predicted_Sentiment_ML' in df.columns:
            y_true = df['True_Sentiment_Category']
            y_pred = df['Predicted_Sentiment_ML']

            # Ensure all labels (Positive, Neutral, Negative) are present in the report
            all_labels = sorted(list(set(y_true.unique()).union(set(y_pred.unique()).union(['Positive', 'Neutral', 'Negative']))))

            st.text("Classification Report:")
            try:
                st.code(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))
            except Exception as e:
                st.error(f"Error generating Classification Report: {e}. Ensure labels are consistent.")
                st.code(classification_report(y_true, y_pred, zero_division=0)) # Fallback without explicit labels

            st.text("Confusion Matrix:")
            try:
                cm = confusion_matrix(y_true, y_pred, labels=all_labels)
                fig_cm, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=all_labels, yticklabels=all_labels)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            except Exception as e:
                st.error(f"Error generating Confusion Matrix: {e}. Check labels and data consistency.")

            st.info("💡 **Insight:** The classification report and confusion matrix provide a detailed breakdown of model performance, highlighting strengths and areas for improvement (e.g., misclassifications between Neutral and Negative reviews).")
        else:
            st.warning("Run sentiment prediction first to see model evaluation.")

    st.markdown("---")

    # --- Row 2: Custom Review Sentiment Prediction & Regression Model ---
    st.subheader("Predict Sentiment & Rating for Custom Reviews")
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        st.write("#### Predict Sentiment for a Custom Review")
        sample_text = st.text_area("Enter a review text:", "This product is okay, but the packaging was damaged.", key="ml_sentiment_text_area")
        if st.button("Predict Sentiment (ML Model)", key="ml_predict_sentiment_btn"):
            if sample_text:
                predicted_s = classification_model.predict([sample_text])[0]
                textblob_s = get_textblob_sentiment_local(sample_text)
                st.success(f"**Predicted Sentiment (ML Model):** {predicted_s}")
                st.info(f"**TextBlob Sentiment (for comparison):** {textblob_s}")
            else:
                st.warning("Please enter some text for prediction.")
        st.info("💡 **Insight:** Test the sentiment model's understanding with your own custom review examples.")


    with col_pred2:
        st.write("#### Predict Product Rating based on Features")
        st.write("Using `regression_model.pkl` and `regression_scaler.pkl` to predict a product's star rating (1-5).")
        st.markdown("##### Input Features for Rating Prediction:")
        prediction_input_method = st.radio("Select input method:", ("Manual Input", "From a Random Review"), horizontal=True, key="regression_input_method")

        text_length_input = 0
        helpfulness_numerator_input = 0
        helpfulness_denominator_input = 0

        if prediction_input_method == "Manual Input":
            col_reg_input1, col_reg_input2 = st.columns(2)
            with col_reg_input1:
                text_length_input = st.number_input("Review Length (characters):", min_value=1, value=100, key="reg_text_len_manual")
            with col_reg_input2:
                helpfulness_numerator_input = st.number_input("Helpfulness Numerator:", min_value=0, value=0, key="reg_help_num_manual")
                helpfulness_denominator_input = st.number_input("Helpfulness Denominator:", min_value=0, value=0, key="reg_help_den_manual")

        elif prediction_input_method == "From a Random Review":
            st.markdown("Click 'Predict Rating' to use characteristics from a randomly sampled review.")
            required_cols = ['ReviewLength', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Summary']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns for random review sampling: {', '.join([col for col in required_cols if col not in df.columns])}. Please check data loading in app2.py.")
                text_length_input = 0
                helpfulness_numerator_input = 0
                helpfulness_denominator_input = 0
            elif not df.empty:
                random_review = df.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]

                st.write(f"**Sampled Review Summary:** *{random_review['Summary']}*")
                st.write(f"**Actual Score:** {random_review['Score']}")

                text_length_input = random_review['ReviewLength']
                helpfulness_numerator_input = random_review['HelpfulnessNumerator']
                helpfulness_denominator_input = random_review['HelpfulnessDenominator']

                st.info(f"Using sampled characteristics: Length={int(text_length_input)}, Helpfulness={int(helpfulness_numerator_input)}/{int(helpfulness_denominator_input)}")

                st.number_input("Review Length (sampled):", value=int(text_length_input), disabled=True, key="reg_text_len_sampled")
                st.number_input("Helpfulness Numerator (sampled):", value=int(helpfulness_numerator_input), disabled=True, key="reg_help_num_sampled")
                st.number_input("Helpfulness Denominator (sampled):", value=int(helpfulness_denominator_input), disabled=True, key="reg_help_den_sampled")
            else:
                st.warning("No data available to sample a random review for regression.")


        # CORRECTED: Pass all three features (ReviewLength, HelpfulnessNumerator, HelpfulnessDenominator)
        if st.button("Predict Rating (ML Model)", key="ml_predict_rating_btn"):
            try:
                input_data_for_reg = np.array([[text_length_input, helpfulness_numerator_input, helpfulness_denominator_input]])
                scaled_features = regression_scaler.transform(input_data_for_reg)
                predicted_score_reg = regression_model.predict(scaled_features)[0]
                predicted_score_reg = max(1, min(5, round(predicted_score_reg)))
                st.success(f"**Predicted Product Rating:** {predicted_score_reg} out of 5 stars")
            except Exception as e:
                st.error(f"Error predicting rating: {e}. Please ensure input features are valid and compatible with the regression model.")

        st.info("💡 **Insight:** This model helps anticipate product performance based on review characteristics, aiding in proactive product management.")

    st.markdown("---")

    # --- Row 3: Review Length vs. Helpfulness & Score Distribution by Length ---
    st.header("2. Review Structure: Length & Impact")
    st.write("Explore how review length influences helpfulness and average review scores.")
    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        st.write("#### Review Length vs. Helpfulness")
        st.info("Does the length of a review impact how helpful users find it? Explore the correlation here.")
        
        sampled_df_len_help = df.sample(n=min(50000, df.shape[0]), random_state=42)
        
        fig_len_help = px.scatter(sampled_df_len_help,
                                    x='ReviewLength',
                                    y='HelpfulnessRatio',
                                    color='Score',
                                    hover_data=['Summary', 'Text', 'Score', 'HelpfulnessRatio'],
                                    title='Review Length vs. Helpfulness Ratio (Sampled Data)',
                                    labels={'ReviewLength': 'Review Length (Characters)', 'HelpfulnessRatio': 'Helpfulness Ratio'},
                                    opacity=0.3,
                                    height=400,
                                    color_continuous_scale=px.colors.sequential.Plasma)
        fig_len_help.update_layout(xaxis_title="Review Length (Characters)",
                                    yaxis_title="Helpfulness Ratio",
                                    coloraxis_colorbar=dict(title="Score"),
                                    margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_len_help, use_container_width=True)
        st.info("💡 **Insight:** While there's no perfect line, very short reviews might lack detail, and excessively long ones could be overwhelming. Aim for clarity and conciseness, especially for top scores.")

    with col_adv2:
        st.write("#### Average Score by Review Length Bins")
        st.info("See if there's an optimal review length for positive (or negative) ratings. Longer reviews might be more detailed for extreme scores.")
        
        df_plot = df.copy()
        df_plot['ReviewLengthBin'] = pd.cut(df_plot['ReviewLength'], bins=[0, 50, 150, 500, 1000, 2000, df_plot['ReviewLength'].max()],
                                         labels=['<50', '50-150', '150-500', '500-1000', '1000-2000', '>2000'],
                                         right=False)
        avg_score_by_len_bin = df_plot.groupby('ReviewLengthBin')['Score'].mean().reset_index()

        fig_score_by_len = px.bar(avg_score_by_len_bin,
                                    x='ReviewLengthBin',
                                    y='Score',
                                    title='Average Score by Review Length Bin',
                                    labels={'ReviewLengthBin': 'Review Length Bin (Characters)', 'Score': 'Average Score'},
                                    color='Score',
                                    color_continuous_scale=px.colors.sequential.Greens,
                                    range_y=[1, 5],
                                    height=400)
        fig_score_by_len.update_layout(xaxis_title="Review Length (Characters)",
                                        yaxis_title="Average Score",
                                        showlegend=False,
                                        margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_score_by_len, use_container_width=True)
        st.info("💡 **Insight:** This reveals if specific review lengths tend to yield higher or lower average scores. There might be a 'sweet spot' for detailed yet positive feedback.")

    st.markdown("---")


    # --- Row 4: Custom Keyword Search & Inter-Feature Correlation ---
    st.header("3. Deeper Text & Feature Analysis")
    st.write("Perform custom searches and understand relationships between numerical features.")
    col_adv3, col_adv4 = st.columns(2)

    with col_adv3:
        st.write("#### Custom Review Search")
        st.info("Find reviews containing specific keywords for targeted feedback analysis. Useful for tracking themes like 'shipping' or 'flavor'.")

        search_query = st.text_input("Enter keywords (e.g., 'organic', 'spicy', 'delivery')", "", key="ml_keyword_search").lower()

        if search_query:
            filtered_df_keyword = df[df['CombinedText'].str.contains(search_query, case=False, na=False)]

            if not filtered_df_keyword.empty:
                st.write(f"Found {len(filtered_df_keyword):,} reviews containing **'{search_query}'**.")
                display_results = filtered_df_keyword.sort_values(by='Time', ascending=False).head(10)[['Time', 'Score', 'HelpfulnessRatio', 'Summary', 'Text']]
                st.dataframe(display_results.style.format({'HelpfulnessRatio': "{:.2%}",
                                                            'Time': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A'
                                                            }),
                                use_container_width=True, height=350, hide_index=True)

                st.info(f"💡 **Insight:** Quickly gather feedback on specific features or recurring issues related to **'{search_query}'**. Sort by helpfulness or score for deeper insights.")
            else:
                st.info(f"No reviews found containing **'{search_query}'**.")
        else:
            st.info("Type keywords above to begin your search.")
        
    with col_adv4:
        st.write("#### Inter-Feature Correlation")
        st.info("Understand how different numerical aspects of reviews relate to each other. Higher absolute correlation values (closer to 1 or -1) indicate stronger relationships.")

        correlation_columns = ['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'HelpfulnessRatio', 'ReviewLength']
        
        existing_corr_cols = [col for col in correlation_columns if col in df.columns]
        if len(existing_corr_cols) > 1:
            corr_matrix = df[existing_corr_cols].corr()

            fig_corr = px.imshow(corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale=px.colors.sequential.RdBu,
                                    range_color=[-1, 1],
                                    title='Correlation Matrix of Review Features',
                                    height=400)
            fig_corr.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info("💡 **Insight:** Positive values (red) indicate features move in the same direction, negative (blue) indicate opposite. E.g., 'Score' and 'ReviewLength' might have a weak correlation, while 'HelpfulnessNumerator' and 'HelpfulnessDenominator' should be strongly positive.")
        else:
            st.warning("Not enough numerical columns available to compute correlation matrix.")

    st.markdown("---")

    # --- Row 5: Word Clouds by Model Predicted Sentiment ---
    st.header("4. Visualizing Sentiment: Word Clouds")
    st.write("Visualizing common themes in reviews based on the model's sentiment classification.")
    
    if not df.empty and 'Predicted_Sentiment_ML' in df.columns:
        df['Cleaned_Text_ML'] = df['Text'].apply(clean_text_local)

        sentiments = df['Predicted_Sentiment_ML'].unique()
        num_sentiments = len(sentiments)
        
        # Create columns dynamically for word clouds, max 3 per row for better layout
        num_cols_per_row = 3
        rows_of_cols = [st.columns(num_cols_per_row) for _ in range((num_sentiments + num_cols_per_row - 1) // num_cols_per_row)]

        current_col_idx = 0
        for sentiment_type in sorted(sentiments):
            with rows_of_cols[current_col_idx // num_cols_per_row][current_col_idx % num_cols_per_row]:
                st.markdown(f"##### {sentiment_type} Reviews Word Cloud")
                text_for_wc = " ".join(df[df['Predicted_Sentiment_ML'] == sentiment_type]['Cleaned_Text_ML'].dropna().astype(str))
                generate_wordcloud_local(text_for_wc, title=f"Top Words in {sentiment_type} Reviews")
            current_col_idx += 1
        st.info("💡 **Insight:** These word clouds reveal the specific language associated with each sentiment, allowing for targeted product improvements or marketing messages.")
    else:
        st.warning("Run sentiment prediction first to generate word clouds.")


    st.markdown("---")

    # --- Row 6: Future Innovations & Recommendations ---
    st.header("🚀 Future Innovations & Recommendations")
    st.markdown("""
    Beyond these insights, we can explore advanced techniques to unlock even more value from your review data:

    * **Topic Modeling (NLP):** Automatically discover and track prevailing themes and discussions within reviews (e.g., using LDA or BERTopic). This can reveal emergent issues or popular features.
    * **Named Entity Recognition (NER):** Precisely identify specific entities (products, brands, locations) in reviews. For example, extracting "organic coffee beans" or "chocolate flavor."
    * **Predictive Models:**
        * **Helpfulness Score Prediction:** Develop models to predict how helpful a new review will be, allowing you to prioritize which reviews to feature or moderate.
        * **Early Sentiment Alerts:** Implement real-time systems to flag highly negative or critical reviews as they come in, enabling immediate customer service intervention or product team awareness.
    * **Business Data Integration:** Combine review data with sales figures, marketing campaign performance, or customer demographics for a holistic business view. Understand how sentiment correlates with sales or marketing spend.
    * **Real-time Analytics Dashboards:** Build live dashboards to monitor customer feedback as it happens, enabling agile responses to emerging trends or issues and providing continuous insights.
    """)