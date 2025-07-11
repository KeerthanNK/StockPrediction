# Importing libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime as dt, timedelta, date
from model import predictionModel  # Ensure this imports correctly

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
])
server = app.server

# App Layout
app.layout = html.Div(
    className="container-fluid p-4",
    children=[
        # Header Section
        html.Div(
            className="jumbotron bg-primary text-white text-center mb-4 shadow rounded",
            children=[
                html.H1("Stock Trend Prediction App 📈", className="display-4"),
                html.P("Analyze and forecast stock trends with ease!", className="lead")
            ]
        ),

        # Input Section
        html.Div(
            className="row mb-4",
            children=[
                # Stock Input
                html.Div(
                    className="col-md-4 mb-3",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Enter Stock Ticker", className="card-title"),
                                dcc.Input(
                                    id='stock_code',
                                    value='',
                                    placeholder='ENTER STOCK TICKER',
                                    type='text',
                                    className='form-control mb-2'
                                ),
                                html.Button('Submit', id='submit-stock', className='btn btn-primary btn-block')
                            ])
                        ])
                    ]
                ),
                # Date Range Input
                html.Div(
                    className="col-md-4 mb-3",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Select Date Range", className="card-title"),
                                dcc.DatePickerRange(
                                    id='date-range',
                                    min_date_allowed=dt(1995, 8, 5),
                                    max_date_allowed=dt.today(),
                                    start_date=date(2020, 1, 1),
                                    end_date=dt.today(),
                                    className='form-control mb-2'
                                )
                            ])
                        ])
                    ]
                ),
                # Forecast Days Input
                html.Div(
                    className="col-md-4 mb-3",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Forecast Days (1-15)", className="card-title"),
                                dcc.Input(
                                    id='n_days',
                                    value='',
                                    type='number',
                                    placeholder='Number of Days',
                                    min=1,
                                    max=15,
                                    className='form-control mb-2'
                                ),
                                html.Button('Forecast', id='Forecast', className='btn btn-success btn-block')
                            ])
                        ])
                    ]
                )
            ]
        ),

        # Button Section
        html.Div(
            className="text-center mb-4",
            children=[
                html.Button('Stock Price', id='stock_price', className='btn btn-info m-2'),
                html.Button('Indicators', id='indicators', n_clicks=0, className='btn btn-warning m-2'),
                html.Button('Yearly Analysis', id='yearly_analysis', n_clicks=0, className='btn btn-secondary m-2')
            ]
        ),

        # Company Info Section
        html.Div(
            className="row mb-4 justify-content-center",
            children=[
                html.Div(
                    className="col-md-8",
                    children=[
                        html.Div(
                            className="alert alert-light text-center shadow-sm",
                            children=[
                                html.Img(id='logo', className='img-fluid mb-2', style={"height": "50px"}),
                                html.H4(id='ticker', className="font-weight-bold"),
                                html.P(id='description', className="text-muted")
                            ]
                        )
                    ]
                )
            ]
        ),

        # Graph Outputs Section
        html.Div(
            className="row",
            children=[
                # Stock Graph
                html.Div(
                    className="col-md-6 mb-4",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Stock Graph", className="card-title text-center"),
                                dcc.Loading(
                                    id='loading2',
                                    type='circle',
                                    children=html.Div([], id='stonks-graph')
                                )
                            ])
                        ])
                    ]
                ),
                # Forecast Graph
                html.Div(
                    className="col-md-6 mb-4",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Forecast Graph", className="card-title text-center"),
                                dcc.Loading(
                                    id='loading3',
                                    type='circle',
                                    children=html.Div([], id='forecast-graph')
                                )
                            ])
                        ])
                    ]
                ),
                # Yearly Analysis Graph
                html.Div(
                    className="col-md-12 mb-4",
                    children=[
                        html.Div(className="card shadow", children=[
                            html.Div(className="card-body", children=[
                                html.H5("Yearly Analysis", className="card-title text-center"),
                                dcc.Loading(
                                    id='loading4',
                                    type='circle',
                                    children=html.Div([], id='yearly-graph')
                                )
                            ])
                        ])
                    ]
                )
            ]
        )
    ]
)
# Callbacks

@app.callback(
    [Output('logo', 'src'), Output('ticker', 'children'), Output('description', 'children')],
    [Input('submit-stock', 'n_clicks')],
    [State('stock_code', 'value')]
)
def update_data(n, stock_code):
    desc = """
    Hey! Enter stock Ticker to get information.

        1. Enter Stock ticker in the input field.
        2. Hit Submit button and wait.
        3. Click Stock Price button or Indicators button to get the stock trend.
        4. Enter the number of days (1-15) to forecast and hit the Forecast button.
        5. Hit the Yearly Analysis button for yearly insights.
    """

    if n == 0 or stock_code == '':
        return 'https://www.linkpicture.com/q/stonks.jpg', '', desc

    try:
        tk = yf.Ticker(stock_code)
        try:
            sinfo = tk.info
        except Exception as info_error:
            return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', f'Error fetching info: {info_error}'

        if not sinfo:
            return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', 'No information available for this ticker.'

        logo_url = sinfo.get('logo_url', 'https://www.linkpicture.com/q/stonks.jpg')
        short_name = sinfo.get('shortName', stock_code)
        business_summary = sinfo.get('longBusinessSummary', 'No description available.')

        if not logo_url or logo_url == '':
            logo_url = 'https://www.linkpicture.com/q/stonks.jpg'

        return logo_url, short_name, business_summary

    except Exception as e:
        return 'https://www.linkpicture.com/q/stonks.jpg', 'Invalid Ticker', f'Error: {str(e)}'

@app.callback(
    Output('stonks-graph', 'children'),
    [
        Input('stock_price', 'n_clicks'),
        Input('indicators', 'n_clicks'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State('stock_code', 'value')]
)
def update_mygraph(n, ind, start, end, stock_code):
    if n == 0 or stock_code == '':
        return ''
    if start is None:
        start = date(2020, 1, 1)
    if end is None:
        end = dt.today()

    df = yf.download(stock_code, start=start, end=end)
    df.reset_index(inplace=True)
    df['ema20'] = df['Close'].rolling(20).mean()
    fig = px.line(df, x='Date', y=['Close'], title='Stock Trend')
    fig.update_traces(line_color='#ef3d3d')

    if ind % 2 != 0:
        fig.add_scatter(x=df['Date'], y=df['ema20'], line=dict(color='blue', width=1), name='EMA20')

    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Close Price")
    return dcc.Graph(figure=fig)

@app.callback(
    Output('forecast-graph', 'children'),
    [Input('Forecast', 'n_clicks'), Input('n_days', 'value')],
    [State('stock_code', 'value')]
)
def forecast(n, n_days, stock_code):
    if n == 0 or stock_code == '' or n_days == '':
        raise PreventUpdate

    try:
        n_days = int(n_days)
        if n_days < 1 or n_days > 15:
            return "Please provide a valid number of days (1-15)."

        fig = predictionModel(n_days + 1, stock_code)
        return dcc.Graph(figure=fig)
    except ValueError:
        return "Please enter a valid number for days."

@app.callback(
    Output('yearly-graph', 'children'),
    [Input('yearly_analysis', 'n_clicks'), Input('n_days', 'value')],
    [State('stock_code', 'value')]
)
def yearly_analysis(n, n_days, stock_code):
    if n == 0 or stock_code == '' or n_days == '':
        raise PreventUpdate

    try:
        n_days = int(n_days)
        fig = analyze_and_visualize_forecast(stock_code, n_days)
        return fig
    except ValueError:
        return "Please enter a valid number for days."

def analyze_and_visualize_forecast(stock_code, n_days):
    df = yf.download(stock_code, start="2011-01-01", end=dt.today())
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    today = dt.today()
    buy_date_str = today.strftime('%m-%d')

    results = []
    for year in range(2011, 2025):
        try:
            buy_date = dt.strptime(f"{year}-{buy_date_str}", "%Y-%m-%d")
            sell_date = buy_date + timedelta(days=n_days)
            if buy_date in df.index and sell_date in df.index:
                buy_price = df.loc[buy_date, 'Close']
                sell_price = df.loc[sell_date, 'Close']
                profit_loss = ((sell_price - buy_price) / buy_price) * 100
                results.append({'Year': year, 'Profit/Loss (%)': profit_loss})
            else:
                results.append({'Year': year, 'Profit/Loss (%)': None})
        except Exception:
            results.append({'Year': year, 'Profit/Loss (%)': None})

    results_df = pd.DataFrame(results)
    results_df['Profit/Loss (%)'].fillna(0, inplace=True)

    positive = len(results_df[results_df['Profit/Loss (%)'] > 0])
    negative = len(results_df[results_df['Profit/Loss (%)'] < 0])
    neutral = len(results_df[results_df['Profit/Loss (%)'] == 0])

    pie_fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[positive, negative, neutral],
        hole=0.3,
        marker=dict(colors=['green', 'red', 'gray'])
    )])

    pie_fig.update_layout(
        title="Profit/Loss Distribution Over the Years"
    )

    bar_fig = go.Figure(
        data=[go.Bar(
            x=results_df['Year'],
            y=results_df['Profit/Loss (%)'],
            marker_color=[
                '#2ca02c' if x > 0 else '#d62728' if x < 0 else '#7f7f7f'
                for x in results_df['Profit/Loss (%)']
            ],
            text=[f"{x:.2f}%" for x in results_df['Profit/Loss (%)']],
            textposition='outside',
            name='Profit/Loss (%)'
        )]
    )
    bar_fig.update_layout(
        title=f"Yearly Analysis: Buy on {buy_date_str}, Sell After {n_days} Days",
        xaxis_title="Year",
        yaxis_title="Returns (%)",
        xaxis=dict(tickmode='linear'),
        yaxis=dict(ticksuffix='%', showgrid=True),
        margin=dict(t=50, b=30, l=30, r=30),
        showlegend=False
    )

    return html.Div([
        dcc.Graph(figure=bar_fig),
        dcc.Graph(figure=pie_fig)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
