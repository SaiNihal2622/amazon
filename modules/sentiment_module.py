import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import io # Used for in-memory image buffer
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Download NLTK resources if not already present
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet, often needed for WordNetLemmatizer

@st.cache_data
def load_data_sentiment():
    try:
        df = pd.read_csv('Reviews.csv')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        df['CombinedText'] = df['Summary'] + ' ' + df['Text']
        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data: {e}")
        st.stop()

@st.cache_data
def clean_text_for_wordcloud(text):
    text = str(text).lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def generate_wordcloud(text_data, max_words=200):
    # Use consistent figsize for matplotlib for better alignment
    fig_height_px = 350 # This should align with the Plotly height
    fig_width_px = 700 # Approx. double the height for landscape

    # Convert pixels to inches for matplotlib figsize
    dpi = 100 # Standard dpi
    figsize_inches = (fig_width_px / dpi, fig_height_px / dpi)

    if not text_data:
        fig, ax = plt.subplots(figsize=figsize_inches)
        ax.text(0.5, 0.5, "No data for word cloud.", ha='center', va='center', fontsize=20, color='gray')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1) # Save to buffer
        plt.close(fig) # Close the figure to free memory
        return buf, "No text data."

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_text_list = []
    for text in text_data:
        cleaned_text = clean_text_for_wordcloud(text)
        tokens = [lemmatizer.lemmatize(word) for word in cleaned_text.split() if word not in stop_words and len(word) > 2]
        cleaned_text_list.extend(tokens)

    long_string = " ".join(cleaned_text_list)
    if not long_string:
        fig, ax = plt.subplots(figsize=figsize_inches)
        ax.text(0.5, 0.5, "No meaningful words after cleaning.", ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return buf, "No meaningful text after cleaning."

    wordcloud = WordCloud(width=fig_width_px, height=fig_height_px, background_color='white',
                          max_words=max_words, collocations=False,
                          colormap='viridis').generate(long_string)

    fig, ax = plt.subplots(figsize=figsize_inches) # Use the defined figsize
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1) # Save to buffer
    plt.close(fig) # Close the figure immediately
    return buf, long_string # Return the buffer

def show_sentiment_dashboard():
    st.title("😊 Sentiment Insights: The Voice of Your Customer")
    st.markdown("---")

    df = load_data_sentiment()

    if df is not None:
        # --- Sentiment Category Breakdown & Trend Over Time (2-in-a-row) ---
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.write("#### 📊 Overall Sentiment Distribution")
            sentiment_map = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
            df['SentimentCategory'] = df['Score'].map(sentiment_map)
            sentiment_counts = df['SentimentCategory'].value_counts()

            sentiment_colors = {
                'Positive': '#4CAF50',
                'Neutral': '#FFD700',
                'Negative': '#FF6347'
            }
            pie_colors = [sentiment_colors[cat] for cat in sentiment_counts.index]

            fig_sentiment = go.Figure(data=[go.Pie(labels=sentiment_counts.index,
                                                   values=sentiment_counts.values,
                                                   pull=[0.05 if cat == 'Negative' else 0 for cat in sentiment_counts.index],
                                                   hole=0.4,
                                                   marker=dict(colors=pie_colors),
                                                   textinfo='percent+label',
                                                   insidetextorientation='radial'
                                                  )])
            fig_sentiment.update_layout(title_text='Proportion of Sentiment Categories',
                                        height=350, # Plotly height
                                        margin=dict(l=20, r=20, t=50, b=20),
                                        showlegend=True,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.info("💡 **Insight:** A high percentage of positive reviews indicates strong customer satisfaction. Focus on understanding the few negative ones for targeted improvements!")

        with col_s2:
            st.write("#### 📈 Sentiment Trend Over Time")
            st.info("Track how the average sentiment (score) has evolved over time. Fluctuations can signal product updates, marketing impacts, or emerging issues.")

            df_time_filtered = df.dropna(subset=['Time'])
            df_time_filtered['YearMonth'] = df_time_filtered['Time'].dt.to_period('M').astype(str)
            sentiment_trend = df_time_filtered.groupby('YearMonth')['Score'].mean().reset_index(name='Average Score')
            sentiment_trend['Date'] = pd.to_datetime(sentiment_trend['YearMonth'])
            sentiment_trend.sort_values('Date', inplace=True)

            time_filter_options = ["All Time", "Last 1 Year", "Last 3 Years", "Last 5 Years"]
            selected_time_filter = st.selectbox("Filter Sentiment Trend:", time_filter_options, key='sentiment_time_filter')

            filtered_sentiment_trend = sentiment_trend.copy()
            if selected_time_filter == "Last 1 Year":
                filtered_sentiment_trend = sentiment_trend[sentiment_trend['Date'] >= (pd.to_datetime(sentiment_trend['Date'].max()) - pd.DateOffset(years=1))]
            elif selected_time_filter == "Last 3 Years":
                filtered_sentiment_trend = sentiment_trend[sentiment_trend['Date'] >= (pd.to_datetime(sentiment_trend['Date'].max()) - pd.DateOffset(years=3))]
            elif selected_time_filter == "Last 5 Years":
                filtered_sentiment_trend = sentiment_trend[sentiment_trend['Date'] >= (pd.to_datetime(sentiment_trend['Date'].max()) - pd.DateOffset(years=5))]

            fig_sentiment_trend = px.line(filtered_sentiment_trend,
                                          x='Date',
                                          y='Average Score',
                                          title='Avg. Sentiment Score Over Time',
                                          labels={'Date': 'Date', 'Average Score': 'Average Score'},
                                          range_y=[1, 5],
                                          height=350, # Plotly height
                                          color_discrete_sequence=[px.colors.qualitative.Plotly[2]])
            fig_sentiment_trend.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sentiment_trend, use_container_width=True)

        st.markdown("---")

        # --- Interactive Word Clouds & Product-Specific Sentiment (2-in-a-row) ---
        col_s3, col_s4 = st.columns(2)

        with col_s3:
            st.write("### ☁️ Uncover Hot Topics with Word Clouds")
            st.write("Explore the most frequent words used in reviews. Adjust the filter to see common themes across different sentiment levels.")

            selected_scores = st.slider(
                "Select Score Range for Word Cloud:",
                min_value=1,
                max_value=5,
                value=(1, 5),
                help="Drag the handles to select a range of star ratings. The word cloud will update to show common words in those reviews."
            )

            filtered_df = df[(df['Score'] >= selected_scores[0]) & (df['Score'] <= selected_scores[1])]

            if filtered_df.shape[0] > 15000:
                sampled_text_data = filtered_df['CombinedText'].sample(n=15000, random_state=42).tolist()
            else:
                sampled_text_data = filtered_df['CombinedText'].tolist()

            if not sampled_text_data:
                st.warning("No reviews found for the selected score range to generate a word cloud.")
            else:
                wordcloud_image_buffer, _ = generate_wordcloud(sampled_text_data)
                # Display the word cloud as an image, allowing Streamlit to handle sizing
                st.image(wordcloud_image_buffer, use_column_width=True) # use_column_width for responsiveness

            st.info("💡 **Insight:** Words like 'delicious', 'great' dominate positive reviews, while 'bad', 'disappointed' appear in negative ones. This directly informs product strengths and areas needing attention!")


        with col_s4:
            st.write("### 🔍 Product-Specific Sentiment Deep Dive")
            st.write("Select a product to view its average rating trend over time. Identify products with improving or declining sentiment.")

            top_products = df['ProductId'].value_counts().nlargest(20).index.tolist()
            selected_product_id = st.selectbox("Select a Product ID:", options=top_products, key='product_sentiment_select')

            if selected_product_id:
                product_df = df[df['ProductId'] == selected_product_id].copy()
                product_df.dropna(subset=['Time'], inplace=True)
                product_df['YearMonth'] = product_df['Time'].dt.to_period('M').astype(str)

                if not product_df.empty:
                    product_sentiment_trend = product_df.groupby('YearMonth')['Score'].mean().reset_index()
                    product_sentiment_trend['Date'] = pd.to_datetime(product_sentiment_trend['YearMonth'])
                    product_sentiment_trend.sort_values('Date', inplace=True)

                    fig_product_sentiment = px.line(product_sentiment_trend,
                                                    x='Date',
                                                    y='Score',
                                                    title=f'Avg. Score Trend for Product: {selected_product_id}',
                                                    labels={'Date': 'Date', 'Score': 'Average Score'},
                                                    range_y=[1, 5],
                                                    height=350, # Set consistent height with other plots
                                                    color_discrete_sequence=[px.colors.qualitative.D3[4]])
                    fig_product_sentiment.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_product_sentiment, use_container_width=True)

                    st.info(f"💡 **Insight:** For **{selected_product_id}**, a declining trend in average score signals a potential issue. An upward trend suggests successful product or service improvements!")
                else:
                    st.warning("No reviews with valid timestamps found for the selected product.")
            else:
                st.info("Please select a product ID from the dropdown to see its sentiment trend.")