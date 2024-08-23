import streamlit as st
import model


# 사이드바 구성
with st.sidebar:
    st.title('💬 2024년 인공지능 산업 최신 동향 Q&A')

# 시작메시지 수정 필요 + 클리어버튼 정의,실행
def clear_chat_history():
    # 1. 메시지 초기화
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you?",
        }
    ]
    # 2. 메모리 초기화 (기록 삭제)
    try:
        st.session_state.memory.clear()
    except:
        pass
st.sidebar.button('Clear', on_click=clear_chat_history)


# 메인페이지 구성
st.title("🤖 2024 AI 산업 동향 Q&A 챗봇")
st.caption("📌 Langchain을 기반으로 한 한국어 인공지능 챗봇입니다.")

# 시작메시지 수정 필요
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("메시지를 입력해 주세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = model.get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)