import streamlit as st
from test import run

# 應用程序的標題
st.title('顯示照片示例')

# 加載圖像
image = st.file_uploader("上傳圖像", type=["jpg", "jpeg", "png"])

# 如果用戶上傳了圖像
if image is not None:
    # 提供临时文件的路径给 run 函数
    white_ratio = run(image)
    
    # 显示结果
    st.header(white_ratio)



