from flask import Flask, jsonify, request
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Load the datasets
sales_data = pd.read_csv(os.path.join('static', 'sales_train_evaluation.csv'))
sell_prices = pd.read_csv(os.path.join('static', 'sell_prices.csv'))
calendar_data = pd.read_csv(os.path.join('static', 'calendar.csv'))

# Preprocess the sales data
sales_data_long = sales_data.melt(id_vars=['id', 'item_id', 'store_id'], var_name='d', value_name='sales')
sales_data_long = sales_data_long.merge(calendar_data[['d', 'date']], on='d', how='left')
sales_data_long['date'] = pd.to_datetime(sales_data_long['date'])

# Merge with sell prices
sales_data_long = sales_data_long.merge(sell_prices, on=['store_id', 'item_id', 'date'], how='left')

@app.route('/forecast', methods=['GET'])
def forecast():
    store_id = request.args.get('store_id', default=1, type=int)
    item_id = request.args.get('item_id', default='FOODS_1', type=str)

    # Filter data for selected store and item
    store_item_data = sales_data_long[(sales_data_long['store_id'] == store_id) & (sales_data_long['item_id'] == item_id)]
    store_item_sales = store_item_data.groupby('date')['sales'].sum()

    # Fit Auto-ARIMA model
    model = auto_arima(store_item_sales, seasonal=True, m=7, stepwise=True, trace=True)

    # Forecast the next 30 days
    forecast_steps = 30
    forecast = model.predict(n_periods=forecast_steps)

    # Plot the forecast
    plt.figure(figsize=(10,6))
    plt.plot(store_item_sales.index, store_item_sales.values, label='Historical Sales')
    forecast_dates = pd.date_range(store_item_sales.index[-1], periods=forecast_steps+1, freq='D')[1:]
    plt.plot(forecast_dates, forecast, label='Forecast')
    plt.legend()

    # Convert plot to image and encode as base64 for React to display
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the forecast data as JSON
    forecast_data = {
        'forecast': forecast.tolist(),
        'forecast_dates': forecast_dates.tolist(),
        'img_b64': img_b64
    }
    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True)

