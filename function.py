from typing import Dict
import streamlit as st

def check_order_status(order_id: str) -> Dict:
    """注文状況を確認する"""
    for order in st.session_state.orders:
        if order["order_id"] == order_id:
            return {
                "order_id": order_id,
                "status": order["status"],
                "product": order["product"],
                "username": order["username"],
                "date": order["date"]
            }
    return {"error": f"注文が見つかりません {order_id}"}
