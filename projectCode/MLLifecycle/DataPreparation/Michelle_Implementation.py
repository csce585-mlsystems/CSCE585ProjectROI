import yfinance

mainDataSet: pd.DataFrame
companies = ["MSFT"]
mainDataSet = yf.ticker(companies[0]) #<-- PN: Make sure to establish the historical period to train neural network on as well!
def dataSetRetriev():
    mainDataSet = pd.read_csv("f{filePath}", get_quote_funcion('debt equity', 'price to book', 'free cash flow','price to earnings'))
    #Could also set different variables for each metric to pass parameters 
    P/E = get_quote_function('price to earnings')
    D/E = get_quote_function('Dividend earnings')
    #or…
    P/B = get_info('Price Book')
    DCF = get_info('Debt equity')
    return

timePeriod = get_data(ticker, start_date = '01/01/2022', end_date = '01/01/2023'  index_as_date = True) # Here, need to ensure that particular time period is set to ensure that the time periods for stock data aligns with stock market data's time period. Refer to yfinance notes for more information. 
# Below function is not needed
#def dataSetRetriev():
    #mainDataSet = pd.read_csv("f{filePath}", "DCF, P/B, D/E, P/E""Insert neccessary params here""") #
    #return

# End of Body that consists of deriving col attributes and reading in .csv file or data file and transforming it into DataFrame

# Body of the functions used to exercise the substrats of value investing. Will dictate the contents of the trainingSet and testSet. 
companies=[Ulta] #<-- Using stocks: Apple, Amazon, Google, Microsoft
