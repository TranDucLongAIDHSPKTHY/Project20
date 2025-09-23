import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import os

# === Thiết lập trang ===
st.set_page_config(
    page_title="Hệ Thống Đề Xuất Phim",
    page_icon="🎬",
    layout="wide"
)

# === CSS tùy chỉnh ===
st.markdown("""
<style>
    .recommendation-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .recommendation-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-3px);
        transition: all 0.3s ease;
    }
    .rating-stars {
        color: #FFD700;
        font-size: 20px;
    }
    .header {
        color: #FF4B4B;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #e04343;
    }
</style>
""", unsafe_allow_html=True)

# === Đường dẫn dữ liệu ===
DATA_PATH = Path(r"D:\Project_20\data\output\RS_save")
MOVIES_FILE = DATA_PATH / "movies_processed.csv"
RATINGS_FILE = DATA_PATH / "ratings_processed.csv"
RECOMMENDATIONS_FILE = DATA_PATH / "recommendations_all_users_msd_k10.pkl"

# === Tải dữ liệu và gợi ý ===
@st.cache_resource
def load_data_and_recommendations():
    try:
        movies = pd.read_csv(MOVIES_FILE)
        ratings = pd.read_csv(RATINGS_FILE)
        with open(RECOMMENDATIONS_FILE, 'rb') as f:
            all_recommendations = pickle.load(f)
        return movies, ratings, all_recommendations
    except Exception as e:
        st.error(f"Không thể tải dữ liệu hoặc gợi ý: {str(e)}")
        return None, None, None

movies, ratings, all_recommendations = load_data_and_recommendations()

# === Hàm lấy gợi ý user ===
def get_recommendations(user_id, all_recommendations, n_recommendations=10):
    recommendations = all_recommendations.get(user_id)
    if not recommendations:
        return None, f"Không tìm thấy gợi ý cho User {user_id}."
    recommendations = recommendations[:n_recommendations]
    formatted = [(raw_iid, movie_title, pred_rating) for raw_iid, movie_title, pred_rating in recommendations]
    return formatted, None

# === Giao diện Streamlit ===
st.title("Hệ Thống Đề Xuất Phim")
st.markdown("### Sử dụng thuật toán KNN (Collaborative Filtering)")

if movies is not None and ratings is not None and all_recommendations is not None:
    # Sidebar
    st.sidebar.header("Cài Đặt Đề Xuất")
    user_ids = sorted(ratings['user_id'].unique())
    selected_user = st.sidebar.selectbox("Chọn User ID:", user_ids, index=0)
    n_recommendations = 10
    st.sidebar.info(f"Số lượng phim đề xuất: **{n_recommendations}**")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Thông Tin User")
    user_ratings = ratings[ratings['user_id'] == selected_user]
    st.sidebar.metric("Số phim đã xem", len(user_ratings))
    st.sidebar.metric("Rating trung bình", f"{user_ratings['rating'].mean():.2f}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Đề Xuất Phim", "Thống Kê", "Giới Thiệu"])

    # === Tab 1: Đề Xuất Phim ===
    with tab1:
        st.header("Đề Xuất Phim Cá Nhân Hóa")
        if st.button("Tạo Đề Xuất Ngay", type="primary"):
            with st.spinner("Đang tải gợi ý..."):
                recommendations, error = get_recommendations(selected_user, all_recommendations, n_recommendations)
                if error:
                    st.error(error)
                else:
                    st.success("Đề xuất thành công!")
                    st.subheader(f"Top {n_recommendations} phim đề xuất cho User {selected_user}")

                    # Hiển thị 2 hàng × 5 phim
                    for row in [recommendations[i:i+5] for i in range(0, len(recommendations), 5)]:
                        cols = st.columns(5)
                        for i, (movie_id, movie_title, pred_rating) in enumerate(row):
                            with cols[i]:
                                stars = int(round(pred_rating))
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{movie_title}</h4>
                                    <p><strong>Rating dự đoán:</strong> 
                                    <span class="rating-stars">{stars}</span> ({pred_rating:.1f}/5.0)</p>
                                    <p><strong>Movie ID:</strong> {movie_id}</p>
                                </div>
                                """, unsafe_allow_html=True)

    # === Tab 2: Thống Kê ===
    with tab2:
        st.header("Thống Kê Hệ Thống")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số Users", len(ratings['user_id'].unique()))
        col2.metric("Tổng số Phim", len(movies))
        col3.metric("Tổng số Ratings", len(ratings))
        if st.checkbox("Hiển thị lịch sử xem"):
            st.subheader(f"Lịch sử xem của User {selected_user}")
            user_history = user_ratings.merge(movies, on='movie_id')
            history_display = user_history[['title', 'rating']].copy()
            history_display.columns = ['Tên phim', 'Rating']
            history_display.index = range(1, len(history_display) + 1)
            st.dataframe(history_display, use_container_width=True)

    # === Tab 3: Giới Thiệu ===
    with tab3:
        st.header("Giới Thiệu Hệ Thống")
        st.markdown("""
        ### Hệ Thống Đề Xuất Phim
        - **Thuật toán:** KNN (Collaborative Filtering, Item-based, MSD, k=10)
        - **Dataset:** MovieLens 100K
        - **Hiển thị:** Chỉ tên phim, rating dự đoán và Movie ID
        """)

else:
    st.error("""
    **Không tìm thấy dữ liệu hoặc gợi ý!**
    Vui lòng kiểm tra các file trong thư mục:
    - movies_processed.csv
    - ratings_processed.csv
    - recommendations_all_users_msd_k10.pkl
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Hệ thống đề xuất phim</strong> - Collaborative Filtering KNN</p>
    <p>Dataset: MovieLens 100K | Algorithm: KNN (MSD, k=10)</p>
    <p>Built with Streamlit | Python | Machine Learning</p>
</div>
""", unsafe_allow_html=True)
