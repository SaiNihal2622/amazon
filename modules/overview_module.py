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
            
            # Insight Metric for Rating Breakdown
            five_star_reviews = score_counts.get(5, 0)
            total_reviews_for_percent = score_counts.sum()
            five_star_percentage = (five_star_reviews / total_reviews_for_percent) * 100 if total_reviews_for_percent > 0 else 0
            st.markdown(f"""
            <div style="background-color: #e0f7fa; border-left: 5px solid #00BCD4; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 1.1em; color: #00796B;">
                    <b>{five_star_percentage:.1f}%</b> of all reviews are <b>5-star ratings</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


            fig_score = px.bar(score_counts,
                               x=score_counts.index,
                               y=score_counts.values,
                               labels={'x': 'Score', 'y': 'Number of Reviews'},
                               title='Distribution of Customer Ratings',
                               color=score_counts.values,
                               color_continuous_scale=px.colors.sequential.Viridis,
                               height=350)
            fig_score.update_layout(xaxis_title="Star Rating", yaxis_title="Number of Reviews", showlegend=False,
                                    margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_score, use_container_width=True)
            st.info("💡 **Insight:** Dominance of 4 & 5-star ratings signals high satisfaction. Critical reviews (1 & 2 stars) are rare but goldmines for improvement!")

        with col_plot2:
            st.write("#### 📈 Review Activity Over Time")
            df_time_filtered = df.dropna(subset=['Time'])
            df_time_filtered['YearMonth'] = df_time_filtered['Time'].dt.to_period('M').astype(str)

            reviews_over_time = df_time_filtered.groupby('YearMonth').size().reset_index(name='Count')
            reviews_over_time['Date'] = pd.to_datetime(reviews_over_time['YearMonth'])
            reviews_over_time.sort_values('Date', inplace=True)

            time_filter_options = ["All Time", "Last 1 Year", "Last 3 Years", "Last 5 Years"]
            selected_time_filter = st.selectbox("Filter Review Activity:", time_filter_options, key='time_filter_activity')

            filtered_reviews_over_time = reviews_over_time.copy()
            time_offset = None
            if selected_time_filter == "Last 1 Year":
                time_offset = pd.DateOffset(years=1)
            elif selected_time_filter == "Last 3 Years":
                time_offset = pd.DateOffset(years=3)
            elif selected_time_filter == "Last 5 Years":
                time_offset = pd.DateOffset(years=5)

            if time_offset:
                filtered_reviews_over_time = reviews_over_time[reviews_over_time['Date'] >= (pd.to_datetime(reviews_over_time['Date'].max()) - time_offset)]
            
            # Insight Metric for Review Activity
            total_reviews_period = filtered_reviews_over_time['Count'].sum()
            period_text = selected_time_filter.lower()
            st.markdown(f"""
            <div style="background-color: #e3f2fd; border-left: 5px solid #2196F3; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 1.1em; color: #1565C0;">
                    <b>{total_reviews_period:,}</b> reviews in the <b>{period_text}</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            fig_time = px.line(filtered_reviews_over_time,
                               x='Date',
                               y='Count',
                               title='Review Volume Trend',
                               labels={'Date': 'Date', 'Count': 'Number of Reviews'},
                               height=350,
                               color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
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

            selected_time_filter_score = st.selectbox("Filter Average Score Trend:", time_filter_options, key='time_filter_score')

            filtered_avg_score_over_time = avg_score_over_time.copy()
            time_offset_score = None
            if selected_time_filter_score == "Last 1 Year":
                time_offset_score = pd.DateOffset(years=1)
            elif selected_time_filter_score == "Last 3 Years":
                time_offset_score = pd.DateOffset(years=3)
            elif selected_time_filter_score == "Last 5 Years":
                time_offset_score = pd.DateOffset(years=5)
            
            if time_offset_score:
                filtered_avg_score_over_time = avg_score_over_time[avg_score_over_time['Date'] >= (pd.to_datetime(avg_score_over_time['Date'].max()) - time_offset_score)]

            # Insight Metric for Average Score Trend
            current_avg_score = filtered_avg_score_over_time['Average Score'].mean() if not filtered_avg_score_over_time.empty else 0
            st.markdown(f"""
            <div style="background-color: #fffde7; border-left: 5px solid #FFEB3B; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 1.1em; color: #FBC02D;">
                    Average score for this period: <b>{current_avg_score:.2f} / 5</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


            fig_avg_score = px.line(filtered_avg_score_over_time,
                                    x='Date',
                                    y='Average Score',
                                    title='Avg. Customer Rating Over Time',
                                    labels={'Date': 'Date', 'Average Score': 'Average Score'},
                                    range_y=[1, 5],
                                    height=350,
                                    color_discrete_sequence=[px.colors.qualitative.Plotly[1]])
            fig_avg_score.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_avg_score, use_container_width=True)
            st.info("💡 **Insight:** Generally stable high ratings. Any dips are critical signals for product quality or service issues. Maintain the excellence!")

        with col_plot4:
            st.write("#### 🤝 Helpfulness Ratio Distribution")
            st.info("The 'Helpfulness Ratio' indicates the proportion of users who found a review helpful.")
            
            # Insight Metric for Helpfulness Ratio
            highly_helpful_reviews = df[df['HelpfulnessRatio'] == 1.0].shape[0]
            total_reviews_with_votes = df[df['HelpfulnessDenominator'] > 0].shape[0]
            highly_helpful_percentage = (highly_helpful_reviews / total_reviews_with_votes) * 100 if total_reviews_with_votes > 0 else 0

            st.markdown(f"""
            <div style="background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; font-size: 1.1em; color: #2E7D32;">
                    <b>{highly_helpful_percentage:.1f}%</b> of reviews are considered <b>highly helpful</b> (ratio of 1.0).
                </p>
            </div>
            """, unsafe_allow_html=True)


            fig_helpfulness = px.histogram(df, x='HelpfulnessRatio',
                                            nbins=50,
                                            title='Distribution of Review Helpfulness',
                                            labels={'HelpfulnessRatio': 'Helpfulness Ratio'},
                                            color_discrete_sequence=[px.colors.qualitative.Plotly[2]],
                                            height=350)
            fig_helpfulness.update_layout(xaxis_title="Helpful Votes / Total Votes",
                                            yaxis_title="Number of Reviews",
                                            margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_helpfulness, use_container_width=True)
            st.info("💡 **Insight:** Most reviews get few votes. The peak at 0, and the tail towards 1.0, reveal highly impactful reviews. Understanding them is key to effective feedback!")

        st.markdown("---")
        st.write("### 🔍 Deeper Dive: What's Next?")
        st.markdown("""
        This overview provides a high-level pulse of customer sentiment and platform activity.
        For more granular insights into <b>specific products, user behaviors, or detailed text analysis</b>,
        navigate to the other sections of this dashboard!
        """)