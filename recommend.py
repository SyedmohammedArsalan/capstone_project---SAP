import streamlit as st
import pandas as pd
import numpy as np

class FashionRecommender:
    def __init__(self):
        self.df = self._load_fashion_dataset()
        
    def _load_fashion_dataset(self):
        # Sample dataset with product images, categories, styles, prices, and descriptions
        data = [
            {
                'item_id': 'PID1001',
                'category': 'Tops',
                'style': 'Casual',
                'price': 29.99,
                'image_url': 'https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?auto=format&fit=crop&w=400&q=80',
                'description': 'Casual cotton t-shirt perfect for everyday wear.'
            },
            {
                'item_id': 'PID1002',
                'category': 'Dresses',
                'style': 'Bohemian',
                'price': 59.99,
                'image_url': 'https://images.unsplash.com/photo-1520975695910-0a7a7a7a7a7a?auto=format&fit=crop&w=400&q=80',
                'description': 'Flowy bohemian dress with floral patterns.'
            },
            {
                'item_id': 'PID1003',
                'category': 'Shoes',
                'style': 'Streetwear',
                'price': 89.99,
                'image_url': 'https://images.unsplash.com/photo-1519744792095-2f2205e87b6f?auto=format&fit=crop&w=400&q=80',
                'description': 'Trendy streetwear sneakers for urban style.'
            },
            {
                'item_id': 'PID1004',
                'category': 'Outerwear',
                'style': 'Professional',
                'price': 120.00,
                'image_url': 'https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?auto=format&fit=crop&w=400&q=80',
                'description': 'Elegant professional blazer for office wear.'
            },
            {
                'item_id': 'PID1005',
                'category': 'Bottoms',
                'style': 'Modern Vintage',
                'price': 45.00,
                'image_url': 'https://images.unsplash.com/photo-1495121605193-b116b5b09a6a?auto=format&fit=crop&w=400&q=80',
                'description': 'Vintage style denim jeans with modern fit.'
            },
            {
                'item_id': 'PID1006',
                'category': 'Accessories',
                'style': 'Active',
                'price': 15.00,
                'image_url': 'https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=400&q=80',
                'description': 'Sporty wristband for active lifestyles.'
            }
        ]
        return pd.DataFrame(data)
    
    def get_recommendations(self, query=None, style=None, price_range=None):
        df_filtered = self.df
        
        if style and style != 'All':
            df_filtered = df_filtered[df_filtered['style'] == style]
        
        if price_range:
            df_filtered = df_filtered[(df_filtered['price'] >= price_range[0]) & (df_filtered['price'] <= price_range[1])]
        
        # For simplicity, return top 6 filtered items
        return df_filtered.head(6)

def style_selector():
    with st.sidebar:
        st.header("🎨 Inclusive Style Preferences")
        with st.expander("FILTERS", expanded=True):
            style = st.selectbox(
                "Preferred Style",
                ['All', 'Streetwear', 'Professional', 'Casual', 'Active', 'Bohemian', 'Modern Vintage'],
                index=0
            )
            price_range = st.slider(
                "Price Range ($)",
                10, 150, (10, 100)
            )
        return {
            'style': style,
            'price_range': price_range
        }

def display_recommendations(recommendations):
    st.markdown(
        """
        <style>
        .scrolling-wrapper {
            overflow-x: auto;
            display: flex;
            flex-wrap: nowrap;
            padding-bottom: 1rem;
            scroll-behavior: smooth;
        }
        .card {
            flex: 0 0 auto;
            width: 220px;
            margin-right: 1rem;
            background: white;
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            border: 1px solid #e9ecef;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }
        .card img {
            width: 100%;
            border-radius: 10px;
        }
        .card h4 {
            margin: 0.5rem 0 0.25rem 0;
            font-size: 1.1rem;
            color: #333;
        }
        .card p {
            margin: 0.25rem 0;
            color: #666;
            font-size: 0.9rem;
        }
        .buy-button {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.4rem 1rem;
            background: #4CAF50;
            color: white;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        .buy-button:hover {
            background: #45a049;
        }
        </style>
        <div class="scrolling-wrapper">
        """
        , unsafe_allow_html=True)

    for _, item in recommendations.iterrows():
        st.markdown(f"""
            <div class="card">
                <img src="{item['image_url']}" alt="{item['item_id']}"/>
                <h4>{item['item_id']}</h4>
                <p><strong>Category:</strong> {item['category']}</p>
                <p><strong>Style:</strong> {item['style']}</p>
                <p><strong>Price:</strong> ${item['price']}</p>
                <p>{item['description']}</p>
                <a href="https://example.com/search?q={item['style']}" target="_blank" class="buy-button">Buy Similar</a>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
