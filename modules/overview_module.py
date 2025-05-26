import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data_overview():
    try:
        df = pd.read_csv('Reviews.csv')
        df.drop(columns=['Id', 'ProfileName'], inplace=True, errors='ignore')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
        df['HelpfulnessRatio'] = df.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data: {e}")
        st.stop()

def show_overview_dashboard():
    st.title("📊 Data Overview: Unveiling the Big Picture")
    st.markdown("---") # Visual separator

    st.write("### 🚀 Key Metrics: Your Snapshot of Success")

    df = load_data_overview()

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Reviews", f"{df.shape[0]:,}", help="Total number of customer reviews analyzed.")
        with col2:
            st.metric("Unique Products", f"{df['ProductId'].nunique():,}", help="Number of distinct products reviewed.")
        with col3:
            st.metric("Unique Customers", f"{df['UserId'].nunique():,}", help="Number of unique customers who left reviews.")
        with col4:
            st.metric("Avg. Product Score", f"{df['Score'].mean():.2f} / 5", help="Overall average star rating across all products.")

        st.markdown("---")

        # --- Plot Row 1: Score Distribution & Review Activity ---
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.write("#### ⭐ Customer Rating Breakdown")
            score_counts = df['Score'].value_counts().sort_index()
            fig_score = px.bar(score_counts,
                               x=score_counts.index,
                               y=score_counts.values,
                               labels={'x': 'Score', 'y': 'Number of Reviews'},
                               title='Distribution of Customer Ratings',
                               color=score_counts.values,
                               color_continuous_scale=px.colors.sequential.Viridis, # Changed color scale
                               height=350) # Smaller plot height
            fig_score.update_layout(xaxis_title="Star Rating", yaxis_title="Number of Reviews", showlegend=False,
                                    margin=dict(l=20, r=20, t=50, b=20)) # Tighter margins
            st.plotly_chart(fig_score, use_container_width=True)
            st.info("💡 **Insight:** Dominance of 4 & 5-star ratings signals high satisfaction. Critical reviews (1 & 2 stars) are rare but goldmines for improvement!")

        with col_plot2:
            st.write("#### 📈 Review Activity Over Time")
            df_time_filtered = df.dropna(subset=['Time'])
            df_time_filtered['YearMonth'] = df_time_filtered['Time'].dt.to_period('M').astype(str)

            reviews_over_time = df_time_filtered.groupby('YearMonth').size().reset_index(name='Count')
            reviews_over_time['Date'] = pd.to_datetime(reviews_over_time['YearMonth'])
            reviews_over_time.sort_values('Date', inplace=True)

            # Add a time filter for interactivity
            time_filter_options = ["All Time", "Last 1 Year", "Last 3 Years", "Last 5 Years"]
            selected_time_filter = st.selectbox("Filter Review Activity:", time_filter_options, key='time_filter_activity')

            filtered_reviews_over_time = reviews_over_time.copy()
            if selected_time_filter == "Last 1 Year":
                filtered_reviews_over_time = reviews_over_time[reviews_over_time['Date'] >= (pd.to_datetime(reviews_over_time['Date'].max()) - pd.DateOffset(years=1))]
            elif selected_time_filter == "Last 3 Years":
                filtered_reviews_over_time = reviews_over_time[reviews_over_time['Date'] >= (pd.to_datetime(reviews_over_time['Date'].max()) - pd.DateOffset(years=3))]
            elif selected_time_filter == "Last 5 Years":
                filtered_reviews_over_time = reviews_over_time[reviews_over_time['Date'] >= (pd.to_datetime(reviews_over_time['Date'].max()) - pd.DateOffset(years=5))]

            fig_time = px.line(filtered_reviews_over_time,
                               x='Date',
                               y='Count',
                               title='Review Volume Trend',
                               labels={'Date': 'Date', 'Count': 'Number of Reviews'},
                               height=350, # Smaller plot height
                               color_discrete_sequence=[px.colors.qualitative.Plotly[0]]) # Consistent line color
            fig_time.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_time, use_container_width=True)
            st.info("💡 **Insight:** Witness the platform's growth! Spikes indicate successful campaigns or product launches. Dips? Time to investigate!")

        st.markdown("---")

        # --- Plot Row 2: Average Score Over Time & (New) Helpfulness Ratio Distribution ---
        col_plot3, col_plot4 = st.columns(2)

        with col_plot3:
            st.write("#### 🎯 Average Score Trend")
            avg_score_over_time = df_time_filtered.groupby('YearMonth')['Score'].mean().reset_index(name='Average Score')
            avg_score_over_time['Date'] = pd.to_datetime(avg_score_over_time['YearMonth'])
            avg_score_over_time.sort_values('Date', inplace=True)

            # Add a time filter for interactivity
            selected_time_filter_score = st.selectbox("Filter Average Score Trend:", time_filter_options, key='time_filter_score')

            filtered_avg_score_over_time = avg_score_over_time.copy()
            if selected_time_filter_score == "Last 1 Year":
                filtered_avg_score_over_time = avg_score_over_time[avg_score_over_time['Date'] >= (pd.to_datetime(avg_score_over_time['Date'].max()) - pd.DateOffset(years=1))]
            elif selected_time_filter_score == "Last 3 Years":
                filtered_avg_score_over_time = avg_score_over_time[avg_score_over_time['Date'] >= (pd.to_datetime(avg_score_over_time['Date'].max()) - pd.DateOffset(years=3))]
            elif selected_time_filter_score == "Last 5 Years":
                filtered_avg_score_over_time = avg_score_over_time[avg_score_over_time['Date'] >= (pd.to_datetime(avg_score_over_time['Date'].max()) - pd.DateOffset(years=5))]


            fig_avg_score = px.line(filtered_avg_score_over_time,
                                    x='Date',
                                    y='Average Score',
                                    title='Avg. Customer Rating Over Time',
                                    labels={'Date': 'Date', 'Average Score': 'Average Score'},
                                    range_y=[1, 5], # Keep y-axis fixed for better comparison
                                    height=350, # Smaller plot height
                                    color_discrete_sequence=[px.colors.qualitative.Plotly[1]]) # Another consistent line color
            fig_avg_score.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_avg_score, use_container_width=True)
            st.info("💡 **Insight:** Generally stable high ratings. Any dips are critical signals for product quality or service issues. Maintain the excellence!")

        with col_plot4:
            st.write("#### 🤝 Helpfulness Ratio Distribution")
            st.info("The 'Helpfulness Ratio' indicates the proportion of users who found a review helpful.")
            fig_helpfulness = px.histogram(df, x='HelpfulnessRatio',
                                           nbins=50,
                                           title='Distribution of Review Helpfulness',
                                           labels={'HelpfulnessRatio': 'Helpfulness Ratio'},
                                           color_discrete_sequence=[px.colors.qualitative.Plotly[2]], # Another consistent color
                                           height=350) # Smaller plot height
            fig_helpfulness.update_layout(xaxis_title="Helpful Votes / Total Votes",
                                          yaxis_title="Number of Reviews",
                                          margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_helpfulness, use_container_width=True)
            st.info("💡 **Insight:** Most reviews get few votes. The peak at 0, and the tail towards 1.0, reveal highly impactful reviews. Understanding them is key to effective feedback!")

        st.markdown("---")
        st.write("### 🔍 Deeper Dive: What's Next?")
        st.markdown("""
        This overview provides a high-level pulse of customer sentiment and platform activity.
        For more granular insights into **specific products, user behaviors, or detailed text analysis**,
        navigate to the other sections of this dashboard!
        """)