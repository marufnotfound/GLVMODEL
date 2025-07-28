
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data inside the app
X_test_report = pd.read_csv("X_test_report1.csv")
importance_df = pd.read_csv("importance_df1.csv")

# Assuming these dataframes are already in your notebook session:
# X_test_report with predicted_glv, actual_glv, is_repeated_guest, is_canceled
# importance_df with features and importance values

st.title("Hotel Booking GLV Model Dashboard")

# Summary stats
st.header("Prediction Summary")
st.write(f"Total test samples: {len(X_test_report)}")
st.write(f"Mean Predicted GLV: {X_test_report['predicted_glv'].mean():.2f}")
st.write(f"Min Predicted GLV: {X_test_report['predicted_glv'].min():.2f}")
st.write(f"Max Predicted GLV: {X_test_report['predicted_glv'].max():.2f}")

# Filter high-value bookings
high_value_threshold = st.slider("Set High-Value GLV Threshold", 0, int(X_test_report['predicted_glv'].max()), 1000)
high_value_bookings = X_test_report[X_test_report['predicted_glv'] > high_value_threshold]

st.write(f"Number of High-Value Bookings (GLV > {high_value_threshold}): {len(high_value_bookings)}")

# Cancellation rate for high-value bookings
cancellation_rate = high_value_bookings['is_canceled'].mean() * 100
st.write(f"Cancellation rate among High-Value Bookings: {cancellation_rate:.2f}%")

# Display sample high-value bookings table
st.subheader("Sample High-Value Bookings")
st.dataframe(high_value_bookings[['predicted_glv', 'actual_glv', 'is_repeated_guest', 'is_canceled']].head(10))

# Feature importance plot
st.header("Feature Importance")
top_n = st.slider("Number of top features to show", 5, 20, 15)
top_features = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title(f"Top {top_n} Features by Importance (Gain)")
st.pyplot(fig)
