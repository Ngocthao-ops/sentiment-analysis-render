import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import os

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

API_URL = os.getenv('API_URL', 'http://localhost:5000/api')

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Phân Tích Cảm Xúc Bình Luận", 
                   className="text-center my-4",
                   style={'color': "#04478A", 'fontWeight': 'bold'}),
            html.P("Sản phẩm DEMO phân tích bình luận Sách Tiki", 
                   className="text-center text-muted mb-4",
                   style={'fontSize': '18px'})
        ])
    ]),
    
    # Input Card
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Nhập đánh giá của bạn", style={'fontWeight': 'bold'}),
                    style={'backgroundColor': "#b0d1f2"}
                ),
                dbc.CardBody([
                    # Text Area
                    dcc.Textarea(
                        id='review-input',
                        placeholder='Ví dụ: Sản phẩm rất tốt, giao hàng nhanh, giá cả hợp lý...',
                        style={
                            'width': '100%',
                            'height': 120,
                            'fontSize': '16px',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'border': '2px solid #e0e0e0',
                            'fontFamily': 'Arial'
                        },
                        value=''
                    ),
                    
                    # Real-time Star Rating Display
                    html.Div([
                        html.H4("Đánh giá số sao:", 
                              style={'fontWeight': 'bold', 'marginTop': '20px', 'marginBottom': '10px'}),
                        
                        # Star Display
                        html.Div(
                            id='star-display',
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'alignItems': 'center',
                                'gap': '20px',
                                'fontSize': '40px',
                                'margin': '20px 0'
                            }
                        ),
                        
                        # Slider
                        dcc.Slider(
                            id='rating-slider',
                            min=1,
                            max=5,
                            step=1,
                            marks={i: str(i) for i in range(1, 6)},
                            className='mb-3'
                        ),
                        
                        html.Div(id='rating-text', 
                                className='text-center',
                                style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#1E90FF'})
                    ]),
                    
                    # Submit Button
                    html.Div([
                        dbc.Button(
                            "Gửi bình luận",
                            id='submit-btn',
                            color="primary",
                            className="mt-3",
                            size="lg",
                            style={'width': '200px', 'fontSize': '18px'}
                        )
                    ], className='text-center')
                ])
            ], style={'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ], width=12)
    ]),
    
    # Loading & Results
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[html.Div(id='result-output')]
        )
    ], className='mt-4'),
    
    
], fluid=True, className="px-4 py-3", style={'maxWidth': '1200px', 'margin': '0 auto'})

# Callback 1: Real-time rating prediction
@app.callback(
    Output('rating-slider', 'value'),
    Output('star-display', 'children'),
    Output('rating-text', 'children'),
    Input('review-input', 'value'),
    Input('rating-slider', 'value'), 
    prevent_initial_call=False
)
def update_realtime_rating(text, slider_value):
    """Cập nhật rating real-time khi user gõ"""
    # Import callback_context
    from dash import callback_context
    
    # Xác định trigger source
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Hàm tạo sao (để tránh lặp code)
    def create_stars(rating):
        stars = []
        for i in range(rating):
            stars.append(html.Span('★', style={'color': '#FFD700', 'fontSize': '40px'}))
        for i in range(5 - rating):
            stars.append(html.Span('★', style={'color': '#ddd', 'fontSize': '40px'}))
        return stars
    
    # Case 1 - User kéo slider
    if trigger_id == 'rating-slider':
        stars = create_stars(slider_value)
        rating_text = html.Span(f"{slider_value} sao", style={'color': '#1E90FF'})
        return slider_value, stars, rating_text
    
    if not text or text.strip() == '':
        # Default state - 3 sao
        stars = create_stars(3)  # ✅ Dùng hàm
        return 3, stars, "3 sao"
    
    try:
        # Gọi API để dự đoán
        response = requests.post(
            f"{API_URL}/predict",
            json={'text': text, 'model': 'logistic'},
            timeout=2
        )
        
        if response.status_code == 200:
            result = response.json()
            rating = result['rating']
            sentiment = result['sentiment']
            
            # Tạo sao theo rating
            stars = []
            for i in range(rating):
                stars.append(html.Span('★', style={'color': '#FFD700'}))
            for i in range(5 - rating):
                stars.append(html.Span('☆', style={'color': '#ddd'}))
            
            # Màu text theo sentiment
            if sentiment == 'Tích cực':
                color = '#28a745'
            elif sentiment == 'Tiêu cực':
                color = '#dc3545'
            else:
                color = '#ffc107'
            
            rating_text = html.Span(f"{rating} sao", style={'color': color})
            
            return rating, stars, rating_text
        
    except:
        pass
    
    # Fallback - giữ nguyên slider hiện tại hoặc default 3
    stars = create_stars(slider_value if slider_value else 3)
    rating_text = html.Span(f"{slider_value if slider_value else 3} sao", 
                        style={'color': '#999'})
    return slider_value if slider_value else 3, stars, rating_text

# Callback 2: Submit và hiển thị kết quả chi tiết
@app.callback(
    Output('result-output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('review-input', 'value'),
    prevent_initial_call=True
)
def submit_review(n_clicks, text):
     
    if not text or text.strip() == '':
        return dbc.Alert("Vui lòng nhập đánh giá!", color="warning")
    
    try:
        # Gọi API compare để lấy kết quả 2 models
        response = requests.post(
            f"{API_URL}/compare",
            json={'text': text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return create_result_table(result)
        else:
            return dbc.Alert(f"Lỗi: {response.json().get('error')}", color="danger")
            
    except requests.exceptions.ConnectionError:
        return dbc.Alert("Không kết nối được API! Hãy chạy Flask API trước.", color="danger")
    except Exception as e:
        return dbc.Alert(f"Lỗi: {str(e)}", color="danger")

def create_result_table(result):
    """Tạo bảng hiển thị kết quả 2 models"""
    
    log_res = result['logistic_regression']
    rf_res = result['random_forest']
    
    # Định nghĩa màu theo sentiment
    def get_color(sentiment):
        if sentiment == 'positive':
            return '#28a745'
        elif sentiment == 'negative':
            return '#dc3545'
        else:
            return '#ffc107'
    
    log_confidence = min(100, max(0, log_res['confidence']))
    rf_confidence = min(100, max(0, rf_res['confidence']))
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Kết Quả Phân Tích", className='mb-0', style={'fontWeight': 'bold', 'color': "#FFFFFF"})
        ], style={'backgroundColor': "#3a8fe3"}),
        dbc.CardBody([
            # Bảng so sánh
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Model", style={'width': '20%', 'fontSize': '16px'}, className='text-center'),
                        html.Th("Cảm xúc", style={'width': '20%', 'fontSize': '16px'}, className='text-center'),
                        html.Th("Độ tự tin", style={'width': '20%', 'fontSize': '16px'}, className='text-center'),
                        html.Th("Thời gian", style={'width': '25%', 'fontSize': '16px'}, className='text-center')
                    ], style={'backgroundColor': "#bbe8f7"})
                ]),
                html.Tbody([
                    # Logistic Regression
                    html.Tr([
                        html.Td([
                            html.Strong("Logistic Regression"),
                        ]),
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
                                style={'height': '25px', 'fontSize': '14px'}
                            )
                        ]),
                        html.Td(
                            html.Span(f"{log_res['prediction_time_ms']} ms", 
                                     style={'fontSize': '14px'})
                        )
                    ]),
                    # Random Forest
                    html.Tr([
                        html.Td([
                            html.Strong("Random Forest"),
                        ]),
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
                                style={'height': '25px', 'fontSize': '14px'}
                            )
                        ]),
                        html.Td(
                            html.Span(f"{rf_res['prediction_time_ms']} ms", 
                                     style={'fontSize': '14px'})
                        )
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, striped=True)
        ])
    ], className='mt-4', style={'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)