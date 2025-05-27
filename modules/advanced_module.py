import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data_advanced():
    try:
        df = pd.read_csv('Reviews.csv')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        df['CombinedText'] = df['Summary'] + ' ' + df['Text']
        df['ReviewLength'] = df['CombinedText'].apply(len)
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')

        df['HelpfulnessRatio'] = df.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
        
        # Ensure HelpfulnessRatio is capped at 1.0 first to correct any data anomalies where numerator > denominator
        df['HelpfulnessRatio'] = df['HelpfulnessRatio'].clip(upper=1.0)
        # Then, apply the specific request: if ratio is 1.0, set it to 0.98
        df.loc[df['HelpfulnessRatio'] == 1.0, 'HelpfulnessRatio'] = 0.98

        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data: {e}")
        st.stop()

def show_advanced_dashboard():
    st.title("🔬 Advanced Analytics: Deeper Correlations & Custom Insights")
    st.markdown("---")

    df = load_data_advanced()

    if df is not None:
        # --- Row 1: Review Length vs. Helpfulness & Score Distribution by Length ---
        st.write("### 📏 Review Structure: Length & Impact")
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            st.write("#### Review Length vs. Helpfulness")
            st.info("Does the length of a review impact how helpful users find it? Explore the correlation here.")
            
            # Sample data for scatter plot to improve performance with large datasets
            sampled_df = df.sample(n=min(50000, df.shape[0]), random_state=42)
            
            fig_len_help = px.scatter(sampled_df,
                                      x='ReviewLength',
                                      y='HelpfulnessRatio',
                                      color='Score', # Color by review score
                                      hover_data=['Summary', 'Text', 'Score', 'HelpfulnessRatio'],
                                      title='Review Length vs. Helpfulness Ratio (Sampled Data)',
                                      labels={'ReviewLength': 'Review Length (Characters)', 'HelpfulnessRatio': 'Helpfulness Ratio'},
                                      opacity=0.3,
                                      height=400, # Consistent height
                                      color_continuous_scale=px.colors.sequential.Plasma) # Aesthetic color scale
            fig_len_help.update_layout(xaxis_title="Review Length (Characters)",
                                        yaxis_title="Helpfulness Ratio",
                                        coloraxis_colorbar=dict(title="Score"),
                                        margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_len_help, use_container_width=True)
            st.info("💡 **Insight:** While there's no perfect linear correlation, this scatter plot often shows that very **short reviews tend to be less helpful**, and extremely long ones can be overwhelming. Reviews with **medium length and high scores** (lighter colors) often cluster in the higher helpfulness ratio ranges, suggesting a sweet spot for detailed yet concise feedback.")


        with col_adv2:
            st.write("#### Average Score by Review Length Bins")
            st.info("See if there's an optimal review length for positive (or negative) ratings. Longer reviews might be more detailed for extreme scores.")
            
            # Create review length bins for better visualization of averages
            df['ReviewLengthBin'] = pd.cut(df['ReviewLength'], bins=[0, 50, 150, 500, 1000, 2000, df['ReviewLength'].max() + 1], # Added +1 to max to ensure max value is included
                                           labels=['<50', '50-150', '150-500', '500-1000', '1000-2000', '>2000'],
                                           right=False, # Includes the left edge, excludes the right
                                           duplicates='drop') # Drop any duplicate bin edges if they occur
            
            # Drop NaN bins that might result from no data in a certain range
            avg_score_by_len_bin = df.groupby('ReviewLengthBin')['Score'].mean().reindex(
                ['<50', '50-150', '150-500', '500-1000', '1000-2000', '>2000'] # Ensure order of bins
            ).reset_index()
            avg_score_by_len_bin.dropna(subset=['Score'], inplace=True) # Remove bins with no data after reindex


            fig_score_by_len = px.bar(avg_score_by_len_bin,
                                       x='ReviewLengthBin',
                                       y='Score',
                                       title='Average Score by Review Length Bin',
                                       labels={'ReviewLengthBin': 'Review Length Bin (Characters)', 'Score': 'Average Score'},
                                       color='Score',
                                       color_continuous_scale=px.colors.sequential.Greens, # Different color scale
                                       range_y=[1, 5], # Fixed score range
                                       height=400) # Consistent height
            fig_score_by_len.update_layout(xaxis_title="Review Length (Characters)",
                                           yaxis_title="Average Score",
                                           showlegend=False,
                                           margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_score_by_len, use_container_width=True)
            
            if not avg_score_by_len_bin.empty:
                max_score_bin = avg_score_by_len_bin.loc[avg_score_by_len_bin['Score'].idxmax()]
                st.info(f"**Analysis:** Reviews falling into the **{max_score_bin['ReviewLengthBin']} characters** bin exhibit the highest average score of **{max_score_bin['Score']:.2f}**. This suggests that within this length, reviews might be providing enough detail to warrant higher ratings, without becoming overly verbose.")
            else:
                st.info("**Analysis:** No data available to show average score by review length bins.")


        st.markdown("---")

        # --- Row 2: Custom Review Search & Interactive Correlation Matrix ---
        st.write("### 🔍 Custom Insights: Search & Relationships")
        col_adv3, col_adv4 = st.columns(2)

        with col_adv3:
            st.write("#### Custom Review Search")
            st.info("Find reviews containing specific keywords for targeted feedback analysis. Useful for tracking themes like 'shipping' or 'flavor'.")

            search_query = st.text_input("Enter keywords (e.g., 'organic', 'spicy', 'delivery')", "")

            if search_query:
                # Filter for reviews with helpfulness denominator > 0 to focus on impactful search results
                search_results = df[df['CombinedText'].str.contains(search_query, case=False, na=False)]

                if not search_results.empty:
                    st.write(f"Found **{search_results.shape[0]:,}** reviews containing **'{search_query}'**.")
                    display_results = search_results.sort_values(by='Time', ascending=False).head(10)[['Time', 'Score', 'HelpfulnessRatio', 'Summary', 'Text']]
                    st.dataframe(display_results.style.format({'HelpfulnessRatio': "{:.2%}", # Format as percentage
                                                                'Time': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A' # Format datetime
                                                                }),
                                 use_container_width=True, height=350, hide_index=True)

                    st.info(f"💡 **Insight:** This powerful search allows you to **rapidly identify feedback trends** related to specific keywords. For example, searching 'packaging' could reveal common complaints or praises, helping product development or logistics teams prioritize improvements. Sorting by helpfulness or score can further pinpoint impactful feedback.")
                else:
                    st.info(f"No reviews found containing **'{search_query}'**. Try a different keyword.")
            else:
                st.info("Type keywords above to begin your search and uncover specific customer feedback.")
        
        with col_adv4:
            st.write("#### Inter-Feature Correlation")
            st.info("Understand how different numerical aspects of reviews relate to each other. Higher absolute correlation values (closer to 1 or -1) indicate stronger relationships.")

            # Calculate correlation matrix for relevant numerical columns
            correlation_columns = ['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'HelpfulnessRatio', 'ReviewLength']
            corr_matrix = df[correlation_columns].corr()

            # Create a heatmap
            fig_corr = px.imshow(corr_matrix,
                                  text_auto=True, # Display correlation values
                                  color_continuous_scale=px.colors.sequential.RdBu, # Red-Blue for positive/negative correlations
                                  range_color=[-1, 1], # Ensure full correlation range
                                  title='Correlation Matrix of Review Features',
                                  height=400)
            fig_corr.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info("💡 **Insight:** This matrix reveals underlying relationships. For instance, you should observe a **strong positive correlation (close to 1)** between `HelpfulnessNumerator` and `HelpfulnessDenominator`, as one is a subset of the other. A **weak or non-existent correlation** between `ReviewLength` and `Score` might suggest that simply writing a long review doesn't guarantee a higher rating, reinforcing the idea that **quality over quantity** matters.")