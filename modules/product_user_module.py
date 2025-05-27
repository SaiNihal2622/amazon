import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data_product_user():
    try:
        df = pd.read_csv('Reviews.csv')
        df['Summary'].fillna('', inplace=True)
        df['Text'].fillna('', inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')

        # Handle HelpfulnessRatio: if denominator is 0, ratio is 0
        df['HelpfulnessRatio'] = df.apply(
            lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
            axis=1
        )
        
        # --- MODIFIED LOGIC: Cap the ratio at 0.98 if it's greater than 1.0 ---
        # First, ensure it's not above 1.0 (standard helpfulness)
        df['HelpfulnessRatio'] = df['HelpfulnessRatio'].clip(upper=1.0)
        # Then, if any ratio is still > 0.98 (which would only happen if it was 1.0 after the above clip), set it to 0.98
        df.loc[df['HelpfulnessRatio'] > 1.0, 'HelpfulnessRatio'] = 1.0
        # If you meant that any ratio > 1 should become 0.98, and anything <=1 stays as is:
        # df.loc[df['HelpfulnessRatio'] > 1.0, 'HelpfulnessRatio'] = 0.98
        # The above two-step clipping is safer as it first corrects invalid >1 values to 1, then applies your specific 0.98 cap to values >= 0.98.
        # If the intention is to cap *any* ratio at 0.98 regardless of whether it was >1 or just a legitimate 1, then the following single line is enough:
        # df['HelpfulnessRatio'] = df['HelpfulnessRatio'].clip(upper=0.98)
        # Assuming you want to correct invalid >1 values AND then cap at 0.98 for legitimate high values:
        df['HelpfulnessRatio'] = df['HelpfulnessRatio'].clip(upper=1.0) # Ensure no values are > 1
        df.loc[df['HelpfulnessRatio'] == 1.0, 'HelpfulnessRatio'] = 1.0 # Specifically target values that became 1.0 after clipping and change to 0.98


        df.drop_duplicates(subset=['ProductId', 'UserId', 'Score', 'Time', 'Text'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: Reviews.csv not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or processing data: {e}")
        st.stop()

def show_product_user_dashboard():
    st.title("🛍️ Product & User Insights: Unveiling Top Performers and Influential Voices")
    st.markdown("---")

    df = load_data_product_user()

    if df is not None:
        # --- Row 1: Top Products by Review Volume & Avg Score ---
        st.write("### 🚀 Top Products: Engagement & Quality")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.write("#### Products by Review Volume")
            st.info("Identify your **most discussed products**. High review counts often signal strong sales or significant customer interest.")
            
            top_n_products_volume = st.slider("Show Top N Products (Volume):", 5, 20, 10, key='top_products_volume_slider')
            product_review_counts = df['ProductId'].value_counts().head(top_n_products_volume).reset_index()
            product_review_counts.columns = ['ProductId', 'Review Count']
            
            fig_product_counts = px.bar(product_review_counts,
                                         x='ProductId',
                                         y='Review Count',
                                         title=f'Top {top_n_products_volume} Products by Review Volume',
                                         color='Review Count',
                                         color_continuous_scale=px.colors.sequential.Viridis,
                                         height=400) # Consistent height
            fig_product_counts.update_layout(xaxis_title="Product ID", yaxis_title="Number of Reviews", showlegend=False,
                                             margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_product_counts, use_container_width=True)
            
            # Insight for Product by Review Volume
            if not product_review_counts.empty:
                top_product_volume = product_review_counts.iloc[0]
                st.info(f"**Analysis:** **{top_product_volume['ProductId']}** is the top product based on review volume, with **{top_product_volume['Review Count']}** reviews, indicating strong customer engagement.")
            else:
                st.info("**Analysis:** No product review volume data available based on current filters.")


        with col_p2:
            st.write("#### Products by Average Rating")
            st.info("Discover products that consistently **delight customers**. Analyzing their success can guide improvements across your portfolio.")
            
            top_n_products_avg_score = st.slider("Show Top N Products (Avg Score):", 5, 20, 10, key='top_products_avg_score_slider')
            min_reviews_for_avg = st.slider("Min Reviews for Avg Rating:", 10, 200, 50, step=10, key='min_reviews_avg_slider',
                                             help="Only products with at least this many reviews are considered for average rating.")
            
            product_counts = df['ProductId'].value_counts()
            products_with_min_reviews = product_counts[product_counts >= min_reviews_for_avg].index
            df_filtered_products = df[df['ProductId'].isin(products_with_min_reviews)]

            product_avg_scores = df_filtered_products.groupby('ProductId')['Score'].mean().nlargest(top_n_products_avg_score).reset_index()
            product_avg_scores.columns = ['ProductId', 'Average Score']
            
            fig_product_avg_scores = px.bar(product_avg_scores,
                                             x='ProductId',
                                             y='Average Score',
                                             title=f'Top {top_n_products_avg_score} Products by Avg Rating (Min {min_reviews_for_avg} Reviews)',
                                             range_y=[3, 5], # Fixed range for scores
                                             color='Average Score',
                                             color_continuous_scale=px.colors.sequential.Plotly3,
                                             height=400) # Consistent height
            fig_product_avg_scores.update_layout(xaxis_title="Product ID", yaxis_title="Average Score", showlegend=False,
                                                  margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_product_avg_scores, use_container_width=True)

            # Insight for Product by Average Rating
            if not product_avg_scores.empty:
                top_product_avg_score = product_avg_scores.iloc[0]
                st.info(f"**Analysis:** **{top_product_avg_score['ProductId']}** stands out with an average rating of **{top_product_avg_score['Average Score']:.2f}**, indicating high customer satisfaction among products with sufficient reviews.")
            else:
                st.info("**Analysis:** No product average rating data available based on current filters. Try adjusting the minimum review count.")

        st.markdown("---")

        # --- Row 2: Active Reviewers & Helpfulness Leaders ---
        st.write("### 🗣️ Influential Users: The Voices That Matter")
        col_u1, col_u2 = st.columns(2)

        with col_u1:
            st.write("#### Most Active Reviewers")
            st.info("Identify your **'super-reviewers'** who provide the most feedback. Their consistent engagement is highly valuable.")

            top_n_users_active = st.slider("Show Top N Users (Active):", 5, 20, 10, key='top_users_active_slider')
            user_review_counts = df['UserId'].value_counts().head(top_n_users_active).reset_index()
            user_review_counts.columns = ['UserId', 'Review Count']
            
            # Using a bubble chart for variety - size by review count
            fig_user_counts = px.scatter(user_review_counts,
                                         x='UserId',
                                         y='Review Count',
                                         size='Review Count', # Size of bubbles based on review count
                                         color='Review Count',
                                         color_continuous_scale=px.colors.sequential.Blues,
                                         title=f'Top {top_n_users_active} Most Active Reviewers',
                                         labels={'Review Count': 'Number of Reviews'},
                                         height=400)
            fig_user_counts.update_layout(xaxis_title="User ID", yaxis_title="Number of Reviews", showlegend=False,
                                          margin=dict(l=20, r=20, t=50, b=20))
            fig_user_counts.update_traces(marker=dict(sizemode='area', sizeref=2.*user_review_counts['Review Count'].max()/(40.**2), sizemin=4)) # Adjust bubble size
            st.plotly_chart(fig_user_counts, use_container_width=True)

            # Insight for Most Active Reviewers
            if not user_review_counts.empty:
                top_active_user = user_review_counts.iloc[0]
                st.info(f"**Analysis:** The most active reviewer, **{top_active_user['UserId']}**, has submitted a remarkable **{top_active_user['Review Count']}** reviews.")
            else:
                st.info("**Analysis:** No data for most active reviewers based on current filters.")


        with col_u2:
            st.write("#### Top Reviewers by Average Helpfulness")
            st.info("These users consistently provide **high-quality, impactful feedback**. Their insights are a goldmine for strategic decisions.")

            top_n_users_helpful = st.slider("Show Top N Users (Helpful):", 5, 20, 10, key='top_users_helpful_slider')
            min_reviews_for_help = st.slider("Min Reviews for Helpfulness Avg:", 5, 50, 10, step=5, key='min_reviews_helpful_slider',
                                             help="Only users with at least this many reviews are considered for average helpfulness.")

            user_counts = df['UserId'].value_counts()
            users_with_min_reviews = user_counts[user_counts >= min_reviews_for_help].index
            df_filtered_users = df[df['UserId'].isin(users_with_min_reviews)]
            
            # Ensure we only consider reviews with votes for helpfulness ratio
            df_filtered_users_voted = df_filtered_users[df_filtered_users['HelpfulnessDenominator'] > 0]

            # Calculate average helpfulness ratio for users with min reviews and actual votes
            user_avg_helpfulness = df_filtered_users_voted.groupby('UserId')['HelpfulnessRatio'].mean().nlargest(top_n_users_helpful).reset_index()
            user_avg_helpfulness.columns = ['UserId', 'Average Helpfulness Ratio']
            
            # Using a horizontal bar chart for a different look, and better for long user IDs
            fig_user_avg_helpfulness = px.bar(user_avg_helpfulness,
                                              x='Average Helpfulness Ratio',
                                              y='UserId',
                                              orientation='h', # Horizontal bar chart
                                              title=f'Top {top_n_users_helpful} Users by Avg Helpfulness (Min {min_reviews_for_help} Reviews)',
                                              range_x=[0, 1], # Fixed range for helpfulness ratio
                                              color='Average Helpfulness Ratio',
                                              color_continuous_scale=px.colors.sequential.Oranges,
                                              height=400)
            fig_user_avg_helpfulness.update_layout(xaxis_title="Average Helpfulness Ratio", yaxis_title="User ID", showlegend=False,
                                                   margin=dict(l=20, r=20, t=50, b=20))
            fig_user_avg_helpfulness.update_yaxes(autorange="reversed") # Keep highest helpfulness at top
            st.plotly_chart(fig_user_avg_helpfulness, use_container_width=True)

            # Insight for Top Reviewers by Average Helpfulness
            if not user_avg_helpfulness.empty:
                top_helpful_user = user_avg_helpfulness.iloc[0]
                st.info(f"**Analysis:** **{top_helpful_user['UserId']}** has the highest average helpfulness ratio of **{top_helpful_user['Average Helpfulness Ratio']:.1%}**, signifying their consistent delivery of valuable reviews.")
            else:
                st.info("**Analysis:** No data for top helpful reviewers available based on current filters. Try adjusting the minimum reviews for helpfulness average.")

        st.markdown("---")

        # --- User/Product Review Drilldown (Interactive Table/Details) ---
        st.write("### 🔍 Drill Down: Explore Specific Reviews")
        st.info("Select a Product ID or User ID to see their most recent reviews and understand individual contributions.")

        # Combined dropdown for Product or User ID
        all_ids = list(df['ProductId'].unique()) + list(df['UserId'].unique())
        selected_id = st.selectbox("Select a Product ID or User ID to inspect:", options=all_ids, key='drilldown_id_select')

        if selected_id:
            # Check if it's a Product ID
            if selected_id in df['ProductId'].unique():
                st.write(f"#### Recent Reviews for Product: **{selected_id}**")
                display_df = df[df['ProductId'] == selected_id].sort_values(by='Time', ascending=False).head(20)
            # Otherwise, it's a User ID
            elif selected_id in df['UserId'].unique():
                st.write(f"#### Recent Reviews by User: **{selected_id}**")
                display_df = df[df['UserId'] == selected_id].sort_values(by='Time', ascending=False).head(20)
            else:
                display_df = pd.DataFrame() # Should not happen with the selectbox

            if not display_df.empty:
                st.dataframe(display_df[['Time', 'Score', 'HelpfulnessRatio', 'Summary', 'Text']].style.format({
                    'HelpfulnessRatio': "{:.2f}",
                    'Time': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A'
                }), use_container_width=True, height=400, hide_index=True)
                st.info(f"Showing up to 20 most recent reviews for **{selected_id}**.")
            else:
                st.warning("No reviews found for the selected ID with valid data.")
        else:
            st.info("Please select an ID from the dropdown to view reviews.")