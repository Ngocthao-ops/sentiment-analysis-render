import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Phân Tích Cảm Xúc (Sentiment Analysis)",
    update_title=None 
)

API_URL = os.getenv('API_URL', 'http://localhost:5000/api')

# Layout
app.layout = html.Div([
    dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Phân Tích Cảm Xúc Bình Luận", 
                   className="text-center my-4",
                   style={'color': "#FFFFFF", 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'}),
            html.P("Sản phẩm DEMO phân tích bình luận Sách Tiki", 
                   className="text-center mb-4",
                   style={'fontSize': '18px', 'color': '#E8F4F8'})
        ])
    ]),
    
    # Input Card
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Nhập đánh giá của bạn", style={'fontWeight': 'bold', 'color': '#FFFFFF'}),
                    style={'backgroundColor': "#4879ea",
                           'borderTopLeftRadius': '15px',
                           'borderTopRightRadius': '15px',
                           'border': 'none'
                    }
                ),
                dbc.CardBody([
                    # Text Area
                    dcc.Textarea(
                        id='review-input',
                        placeholder='Sách hay, giao hàng nhanh, giá cả hợp lý...',
                        style={
                            'width': '100%',
                            'height': 120,
                            'fontSize': '16px',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'border': '2px solid #64b5f6',
                            'fontFamily': 'Arial',
                            'backgroundColor': '#FFFFFF'
                        },
                        value=''
                    ),
                    
                    # Real-time Star Rating Display
                    html.Div([
                        html.H4("Đánh giá số sao:", 
                              style={'fontWeight': 'bold', 'marginTop': '20px', 'marginBottom': '15px', 'color': '#1565c0'}),
                        
                        # Clickable Star Display
                        html.Div(
                            id='star-display',
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'alignItems': 'center',
                                'gap': '10px',
                                'fontSize': '50px',
                                'margin': '20px 0',
                                'cursor': 'pointer'
                            }
                        ),
                        
                        html.Div(id='rating-text', 
                                className='text-center',
                                style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#1976d2', 'marginBottom': '10px'})
                    ]),
                    
                    # Submit Button
                    html.Div([
                        dbc.Button(
                            "Gửi bình luận",
                            id='submit-btn',
                            color="primary",
                            className="mt-3",
                            size="lg",
                            style={
                                'width': '220px', 
                                'fontSize': '18px',
                                'backgroundColor': '#1976d2',
                                'border': 'none',
                                'boxShadow': '0 4px 6px rgba(33, 150, 243, 0.4)',
                                'fontWeight': 'bold'
                            }
                        )
                    ], className='text-center')
                ], style={'backgroundColor': '#FFFFFF'})
            ], style={'boxShadow': '0 8px 16px rgba(0,0,0,0.2)', 'border': 'none', 'borderRadius': '15px', 'overflow': 'hidden'})
        ], width=12)
    ]),
    
    # Loading & Results
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[html.Div(id='result-output')],
            color="#1976d2"
        )
    ], className='mt-4'),
    
    # Hidden store for rating
    dcc.Store(id='current-rating', data=5)
    
], fluid=True, className="px-4 py-5", style={
    'maxWidth': '1200px', 
    'margin': '0 auto',
    'position': 'relative',
    'zIndex': 1
})
], style={
    'minHeight': '100vh',
    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%)',
    'backgroundSize': '400% 400%',
    'animation': 'gradient 15s ease infinite',
    'backgroundAttachment': 'fixed'
})

# Add CSS for gradient animation
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
            }
            #react-entry-point {
                min-height: 100vh;
            }
            .star-clickable:hover {
                transform: scale(1.2);
                filter: brightness(1.2);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback 1: Real-time rating prediction with clickable stars
@app.callback(
    Output('star-display', 'children'),
    Output('rating-text', 'children'),
    Output('current-rating', 'data'),
    Input('review-input', 'value'),
    Input({'type': 'star', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State('current-rating', 'data'),
    prevent_initial_call=False
)
def update_realtime_rating(text, star_clicks, current_rating):
    """Cập nhật rating real-time khi user gõ hoặc click sao"""
    from dash import callback_context
    import dash
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Hàm tạo sao có thể click
    def create_clickable_stars(rating):
        stars = []
        
        for i in range(1, 6):
            star = html.Span(
                '★',
                id={'type': 'star', 'index': i},
                n_clicks=0,
                style={
                    'color': '#FFD700' if i <= rating else '#ddd',
                    'fontSize': '50px',
                    'cursor': 'pointer',
                    'transition': 'all 0.2s ease',
                    'display': 'inline-block',
                    'margin': '0 5px',
                    'userSelect': 'none'
                },
                className='star-clickable'
            )
            stars.append(star)
        return stars
    
    # Xử lý click vào sao
    if trigger_id and 'star' in trigger_id:
        import json
        try:
            star_data = json.loads(trigger_id)
            clicked_star = star_data['index']
            
            # Cập nhật rating dựa trên sao được click
            rating = clicked_star
            stars = create_clickable_stars(rating)
                       
            sentiment_colors = {
                1: '#f44336',
                2: '#ff9800',
                3: '#ffc107',
                4: '#8bc34a',
                5: '#4CAF50'
            }
            
            rating_text = html.Span(
                f"{rating} sao", 
                style={'color': sentiment_colors[rating], 'fontWeight': 'bold'}
            )
            
            return stars, rating_text, rating
        except:
            pass
    
    # Hàm chuyển đổi prediction (-1, 0, 1) sang rating (1-5)
    def prediction_to_rating(prediction, confidence):
        """
        Chuyển đổi prediction thành rating 1-5 dựa trên class và confidence
        """
        if prediction == 1:  # Positive
            # Confidence cao (>80%) = 5 sao, thấp hơn = 4 sao
            return 5 if confidence > 80 else 4
        elif prediction == 0:  # Neutral
            return 3
        else:  # Negative (prediction == -1)
            # Confidence cao (>80%) = 1 sao, thấp hơn = 2 sao
            return 1 if confidence > 80 else 2
    
    # Default: 5 sao
    if not text or text.strip() == '':
        rating = 5
        stars = create_clickable_stars(rating)
        rating_text = html.Span("5 sao", style={'color': '#4CAF50'})
        return stars, rating_text, rating
    
    # Dự đoán nhanh chỉ với Logistic Regression
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={'text': text, 'model': 'logistic'},
            timeout=1  
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']  # -1, 0, 1
            confidence = result['confidence']  # 0-100
            
            # Chuyển đổi sang rating 1-5
            rating = prediction_to_rating(prediction, confidence)
            
            # Tạo sao
            stars = create_clickable_stars(rating)
            
            # Text với màu tương ứng
            sentiment_colors = {
                1: '#f44336',  # Đỏ - Rất tệ
                2: '#ff9800',  # Cam - Tệ
                3: '#ffc107',  # Vàng - Trung bình
                4: '#8bc34a',  # Xanh lá nhạt - Tốt
                5: '#4CAF50'   # Xanh lá đậm - Tuyệt vời
            }
            
            
            rating_text = html.Span(
                f"{rating} sao", 
                style={'color': sentiment_colors[rating], 'fontWeight': 'bold'}
            )
            
            return stars, rating_text, rating
        
    except:
        pass
    
    # Fallback - giữ nguyên rating hiện tại hoặc 5 sao
    rating = current_rating if current_rating else 5
    stars = create_clickable_stars(rating)
     
    rating_text = html.Span(
        f"{rating} sao", 
        style={'color': '#999'}
    )
    return stars, rating_text, rating

# Callback 2: Submit và hiển thị kết quả chi tiết
@app.callback(
    Output('result-output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('review-input', 'value'),
    prevent_initial_call=True
)
def submit_review(n_clicks, text):
    if not text or text.strip() == '':
        return dbc.Alert("Vui lòng nhập đánh giá!", color="warning", style={'borderRadius': '8px'})
    
    try:
        response = requests.post(
            f"{API_URL}/compare",
            json={'text': text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return create_result_table(result)
        else:
            return dbc.Alert(f"Lỗi: {response.json().get('error')}", color="danger")
            
    except requests.exceptions.Timeout:
        return dbc.Alert("Vui lòng thử lại.", color="warning")
    except requests.exceptions.ConnectionError:
        return dbc.Alert("Không kết nối được API! Hãy chạy Flask API trước.", color="danger")
    except Exception as e:
        return dbc.Alert(f"Lỗi: {str(e)}", color="danger")

def create_result_table(result):
    """Tạo bảng hiển thị kết quả 3 models"""
    
    log_res = result['logistic_regression']
    rf_res = result['random_forest']
    cnn_res = result.get('cnn_lstm')
    
    def get_color(sentiment):
        if sentiment == 'positive':
            return '#4CAF50'
        elif sentiment == 'negative':
            return '#f44336'
        else:
            return '#ff9800'
    
    log_confidence = min(100, max(0, log_res['confidence']))
    rf_confidence = min(100, max(0, rf_res['confidence']))
    
    table_rows = [
        # Logistic Regression
        html.Tr([
            html.Td(html.Strong("Logistic Regression"), style={'fontSize': '15px'}),
            html.Td(
                html.Span(log_res['sentiment'], 
                         style={'color': get_color(log_res['sentiment_en']), 
                               'fontWeight': 'bold', 'fontSize': '16px'})
            ),
            html.Td([
                dbc.Progress(
                    value=log_confidence,
                    label=f"{log_confidence:.2f}%",
                    color='success' if log_confidence > 75 else 'warning',
                    style={'height': '28px', 'fontSize': '14px'}
                )
            ]),
            html.Td(
                html.Span(f"{log_res['prediction_time_ms']} ms", 
                         style={'fontSize': '14px', 'color': '#666'})
            )
        ]),
        # Random Forest
        html.Tr([
            html.Td(html.Strong("Random Forest"), style={'fontSize': '15px'}),
            html.Td(
                html.Span(rf_res['sentiment'], 
                         style={'color': get_color(rf_res['sentiment_en']), 
                               'fontWeight': 'bold', 'fontSize': '16px'})
            ),
            html.Td([
                dbc.Progress(
                    value=rf_confidence,
                    label=f"{rf_confidence:.2f}%",
                    color='success' if rf_confidence > 75 else 'warning',
                    style={'height': '28px', 'fontSize': '14px'}
                )
            ]),
            html.Td(
                html.Span(f"{rf_res['prediction_time_ms']} ms", 
                         style={'fontSize': '14px', 'color': '#666'})
            )
        ])
    ]
    
    # CNN-LSTM
    if cnn_res:
        cnn_confidence = min(100, max(0, cnn_res['confidence']))
        table_rows.append(
            html.Tr([
                html.Td(html.Strong("CNN-LSTM"), style={'fontSize': '15px'}),
                html.Td(
                    html.Span(cnn_res['sentiment'], 
                             style={'color': get_color(cnn_res['sentiment_en']), 
                                   'fontWeight': 'bold', 'fontSize': '16px'})
                ),
                html.Td([
                    dbc.Progress(
                        value=cnn_confidence,
                        label=f"{cnn_confidence:.2f}%",
                        color='success' if cnn_confidence > 75 else 'warning',
                        style={'height': '28px', 'fontSize': '14px'}
                    )
                ]),
                html.Td(
                    html.Span(f"{cnn_res['prediction_time_ms']} ms", 
                             style={'fontSize': '14px', 'color': '#666'})
                )
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Kết Quả Phân Tích", className='mb-0', style={'fontWeight': 'bold', 'color': "#FFFFFF"})
        ], style={'backgroundColor': "#4879ea"}),
        dbc.CardBody([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Model", style={'width': '25%', 'fontSize': '16px', 'color': '#1565c0'}, className='text-center'),
                        html.Th("Cảm xúc", style={'width': '20%', 'fontSize': '16px', 'color': '#1565c0'}, className='text-center'),
                        html.Th("Độ tự tin", style={'width': '30%', 'fontSize': '16px', 'color': '#1565c0'}, className='text-center'),
                        html.Th("Thời gian", style={'width': '25%', 'fontSize': '16px', 'color': '#1565c0'}, className='text-center')
                    ], style={'backgroundColor': "#e3f2fd"})
                ]),
                html.Tbody(table_rows)
            ], bordered=True, hover=True, responsive=True, striped=True, style={'backgroundColor': '#FFFFFF'})
        ], style={'backgroundColor': '#FFFFFF'})
    ], className='mt-4', style={'boxShadow': '0 8px 16px rgba(0,0,0,0.2)', 'border': 'none', 'borderRadius': '15px', 'overflow': 'hidden'})

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)