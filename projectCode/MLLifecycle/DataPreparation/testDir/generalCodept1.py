# NOTE: Will have to run this code in the MLLifecycle. Particularly here: /c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode
# Psuedocode for creating general version of taking in a dataset and then creating a training set out of it. 

# Will have <num of substrats taht value investing contains> functions for trainingSet-TestingSet pairs  


# Necessary imports
import pandas as pd;
import numpy as np;
import yfinance as yf; #<-- Needed to access Yahoo Finance Dataset(s)
# Below is needed to get the real-time data
from datetime import datetime
# End of Necessary imports

# Body that consists of deriving col attributes and reading in .csv file or data file and transforming it into DataFrame
filePath: str
mainDataSet: pd.DataFrame
companies = ["MSFT", "AMZN", "AAPL", "GOOGL"]
mainDataSet = yf.Ticker(companies[0]) #<-- PN: Make sure to establish the historical period to train neural network on as well!
timePeriod = "3mo" # Here, need to ensure that particular time period is set to ensure that the time periods for stock data aligns with stock market data's time period. Refer to yfinance notes for more ifnromation. 
timeInterval = "1mo" # Here, this ensure sthat each row references a month between the time period aforementioned. 
firstDate = "2025-08-01" #-- NOTE: These two dates needa be based on recent months. [complete!]
endDate = "2025-10-01" 
arrayReffingDesiredTimeInterval = []
# Below function is not needed
#def dataSetRetriev():
    #mainDataSet = pd.read_csv("f{filePath}", "Insert neccessary params here") #
    #return

# End of Body that consists of deriving col attributes and reading in .csv file or data file and transforming it into DataFrame

# Body of the functions used to exercise the substrats of value investing. Will dictate the contents of the trainingSet and testSet. 
companies=["Apple", "Amazon", "Google", "Microsoft"] #<-- Using stocks: Apple, Amazon, Google, Microsoft
resultantDataFrame = pd.DataFrame()
resultantDataFrame["Company"] = pd.Series(np.empty(4)) #<-- Initalizies Company Column. 
secondaryDataFrame = mainDataSet.balance_sheet  #<-- Used to obtain Net Tangible Assets Column.
# ^^ Will change name soon
secondaryDataFrame.reset_index() #<-- Resets index so row-tuples can be queried using regular indices, reducing confusion. 

def func1(company, companyIndex = 0):
    # Strat 1: 1) A value stock should have P/B ratio of 1.0 or lower, 2) **The Price-to-earnings (P/E) ratio** should be less than 40% of the stock's highest P/E over the previous five years. 3) Look for a share price that is less than 67% of the tangible per-share book value, AND less than 67% of the company's net current asset value (NCAV), 4) A company's total book value should be greater than its total debt, 5) A comapny's total debt SHOULD NOT exceed **twice the NCAV, and total current liabilities and long-term debt should NOT be greater than the firm's total stockholer equity** 
    # Psuedosteps: a) Creating a python list of independent variables, b) Using the list of indep variables to create Series, c) Copying the respective series to the column ids required to support model's decisions, d) Removing independent vars from the original dataframe , e) Load the resultant dataset to the testSet file[or create a new one? Go into more detail about this later], f) Use an algorithm that applies the order of optimality based on the strategy to columns on dataframe and then Write the resultant dataframe to testSet file. 

   # Body of testing for TEMPORARY PURPOSES!
   # NOTE: Make sure below reflects updated version above!
   companyIndex = 0
   resultantDataFrame = pd.DataFrame()
   resultantDataFrame["Company"] = pd.Series(np.empty(4)) #<-- Initalizies Company Column. 
   secondaryDataFrame = mainDataSet.quarterly_balance_sheet.T  #<-- Used to obtain Net Tangible Assets Column.
   filePath: str
   mainDataSet: pd.DataFrame
   companies = ["MSFT", "AMZN", "AAPL", "GOOGL"]
   company = companies[0]
   mainDataSet = yf.Ticker(companies[companyIndex]) #<-- PN: Make sure to establish the historical period to train neural network on as well!
   timePeriod = "3mo" # Here, need to ensure that particular time period is set to ensure that the time periods for stock data aligns with stock market data's time period. Refer to yfinance notes for more ifnromation. 
   timePeriod2 = "" #<-- will be needed to address condition #2
   timeInterval = "1mo"
   firstDate = "2025-09-01" 
   endDate = "2025-10-01" 
   arrayReffingDesiredTimeInterval = [firstDate,endDate]

   # end of Body of testing for Temporary Purposes!
   #companies=[]
   # NOTE: Decided to take the high col. In future, will do it another way. Thus high == price.
   indepVariables: str = ["High","Tangible Book Value","Retained Earnings","Net Tangible Assets", "Total Debt" ] #<-- When filling in indepVars, make sure to refer to contents of the ticker object provided by yfinance. FOr more info, refer to this link and filepath respectively: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html#yfinance.Ticker and /c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/asyncNotes/lecture3.md.[UPDATE: After doing researhc, found that the following obj attrs coantin essential info for value investing: a) `balance_sheet`, b) `cash_flow`, c) `financials`, d) `history`. Prepending `quarterly_` to the aforemtneioned attrs allows us to get the quarterly basis, "allowing for more granular analysis". [update: history may not work. part 2) Finished porting evrything. Now need to sift through indices to see which one's I'll need. Also, when using these, make sure to transpose the dataframes first such that row-tupels are created instead of col-tuples]
   # PN: May be wise to do this process iteratively via a for loop since each company will have thier respective values. OR, it can possibly be done using shortcuts provided by pandas and numpy. part 2) Heavily believe that this shoudl be implemnted via for loop to go through each company and add the rows iteratively. 

# NOTE: Retained earnings can only be found on balance sheet. 
   DepVariables = ["Company", "P/B", "P/E", "Share Price", "NCAV", "Company's Debt"]
   resultantDataFrame = pd.DataFrame()
   #realTimeDataFrameFrmTing: pd.DataFrame() = mainDataSet.history(period=timePeriod, interval=timeInterval).tz_localize(None).loc[arrayReffingDesiredTimeInterval[0]:arrayReffingDesiredTimeInterval[1], :]
   realTimeDataFrameFrmTing: pd.DataFrame() = mainDataSet.history(period=timePeriod, interval=timeInterval).tz_localize(None).loc[0:2]
   companyBalanceSheet = mainDataSet.quarterly_balance_sheet.T.loc[arrayReffingDesiredTimeInterval[0]:arrayReffingDesiredTimeInterval[1], :] #<-- Transposed original to have row-tupls instead of column-tuples
   # companyBalanceSheet = companyBalanceSheet[companyBalanceSheet.index != "2024-03-31"] #<-- Removes index that has NaN vals. 
   companyFinancials = yf.Ticker(f"{companies[companyIndex]}").quarterly_financials.T.loc[arrayReffingDesiredTimeInterval[0]:arrayReffingDesiredTimeInterval[1], :] #<-- PN: May need to query for periodic date based on `timePeriod` variable. 
   companyFinancials.reset_index() #<-- Resets index so row-tuples can be queried using regular indices, reducing confusion.
   # Body of assigning new columns to dataframe 
   resultantDataFrame.loc[companyIndex,"Company"] = company #<-- Adds each company's tuple
   resultantDataFrame["P/B"] = realTimeDataFrameFrmTing[indepVariables[0]]/companyBalanceSheet[indepVariables[1]] #<- NOTE: THis is proding an err b/c both need to be quarterly![complete, only problem is the following error. Refer to the search query here for mroe nifromation: `start chrome "? TypeError: Cannot join tz-naive with tz-aware DatetimeIndex"`

   return #<-- Temporary breakpoint, to fix above. 
   # TESTING: a) Code runs properly up until this point![UPDATE: TO understand what I need to do to fix errs, will continue learning pandas. Once finished learning pandas, then I will use thigns learned to get assignment(s) complete]
   resultantDataFrame["P/E"] = realTimeDataFrameFrmTing[indepVariables[0]]/companyBalanceSheet[indepVariables[2]]
   resultantDataFrame["Share Price"] = realTimeDataFrameFrmTing["High"]
   # resultantDataFrame["Share Price"] = realTimeDataFrameFrmTing[indepVariables[4]]*companyFinancials["Basic Average Shares"] #<-- Here, used following formula to get Share Price: Closed price * Outstanding Shares. [NOTE: Not sure if this is correct. Using realtimeDataFrame instead]
   # NOTE: To get the above, will need to use the alpha vantage python api. Link to documentation is here: https://www.alphavantage.co/documentation/. [also, need to ensure that time periods for resp api calls are the SAME]
   resultantDataFrame["NCAV"] = secondaryDataFrame[indepVariables[3]]#<-- PN: The closest one to this seems like Net Tangible Assets. Refer to /c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/DataPreparation/outputFromAttrs.txt for reference. [complete, it has been set]
   resultantDataFrame["Company's Debt"] = secondaryDataFrame[indepVariables[4]] # [complete, it has been set] 
   # (cont here)


   # End of assigning new columns to dataframe 

   # Body of adding column to dataframe to create trainingSet  
   # Can call function that uses the conditions to determine applicability[need to figure out how to implement order of optimality][PN: May be helpful to do it manually. I think for sake of speed, I may use benchmark data specifically instead of doing this manually. UPDATE: Need to create trainingSet manually, and then create test set based on benchmark data for comparison purposes] 
   scoreTable = {}
   def applyingConditions(company):
       # Good psuedosteps to start with: 1) Use scoring system
       # 1) Scoring system: a) Since a stock has n optimal features, we can have the max score be n. Thus, each company needs to be tested to see if they get the maximum score. Then, the score mus tbe associated with that particular company[think using a dict here where key is company and value is score would be good]. 
       # (cont writing code here)
       # 1)
       # Using match statement that increments score. Each case must be visited, and score is incremented when needed: 
       # Using a while loop here to ensure that every case is progressed through
       # PN: May make sense to make everything below from float to a np.array and ensure that
       # the sum of the array is > 0 which is an indication that it is true. OR I can use averages
       # for determinations based on period. If period is 1d then averages aren't needed. [complete]
       numConditionsIterated = 0;
       while(numConditionsIterated < 4):
           match numConditionsIterated:
                case 0:
                    # (body of code that determine cond1)
                    p_bratio_of_company: np.array = resultantDataFrame["P/B"]
                    # Then, here I will need to use 
                    # (end of body of code that determine cond1)
                    boolExp: bool = (p_bratio_of_company <= 1).all().sum() > 0
                    score = score + 1 if boolExp else score
                case 1:
                    # (body of code that determine cond2)
                    p_bratio_of_company_for_prev_five_years: np.array 
        # Then, here I will need to use 
                    boolExp: bool = (p_bratio_of_company_for_prev_five_years < 0.4).all().sum() == 5
                    # ^^ Above uses a numpy array and checks if all p/b ratios of compnay from prev five years is less than 0.4
                    # (end of body of code that determine cond2)
                    boolExp: bool
                    score = score + 1 if boolExp else score
                case 2:
                    # (body of code that determine cond3)
                    share_price_of_company: np.array
                    tangible_per_share_book_val: np.array
                    share_price_of_company: np.array
                    ncav_of_company: np.array
                    boolExp: bool = (share_price_of_company < (0.67 * tangible_per_share_book_val)).all().sum() > 0 and (share_price_of_company < (0.67 * ncav_of_company)).all().sum() > 0
                    # (end of body of code that determine cond3)
                    score = score + 1 if boolExp else score
                case 3:
                    # (body of code that determine cond3)
                    # (end of body of code that determine cond3)
                    company_book_val: np.array
                    company_total_debt: np.array
                    boolExp: bool = company_book_val > company_total_debt
                    score = score + 1 if boolExp else score
                case 4:
                    # (body of code that determine cond4)
                    company_book_val: np.array
                    company_total_debt: np.array
                    company_total_liabilities: np.array
                    ncav_of_company: np.array
                    company_total_stockholder_equity: np.array

                    # (end of body of code that determine cond4)
                    boolExp: bool = company_total_debt > (2 * ncav_of_company) and ( company_total_liabilities > company_total_liabilities and company_total_debt > company_total_stockholder_equity)
                    score = score + 1 if boolExp else score
       scoreTable[company] = score #<-- Sends score for future processing. 
       # end of 1)
# End of using a while loop here to ensure that every case is progressed through
       return 2
   # After getting score table, will assign score to resp companies: 
   k = len(scoreTable)
   resultantDataFrame["Optimality"] = pd.Series()
   for i in range(k):
       resultantDataFrame.at[companies[i],"Optimality"] = scoreTable[companies[i]]
    # At this point, companies' will have their resp optimalities added. 




   # End of Body of adding column to dataframe to create trainingSet  
   # At this point, the for loop should end. 
   # Body of writing resultant Dataframe to testSet
   filepathToTestSetDir: str = "MLLifecycle/ModelDevelopment/TestSets"
   resultantDataFrame.to_csv(f"{filepathToTestSetDir}/testSetStrat1.csv")
   # End of Body of writing resultant Dataframe to testSet
   # Body of adding column to dataframe to create trainingSet  
   
   # End of body of adding column to dataframe to create trainingSet  
   # Body of writing dataframe to file for trainingSet1. 
   filepathToTrainingSetDir: str = "MLLifecycle/ModelDevelopment/TrainingSets"
   resultantDataFrame.to_csv(f"{filepathToTrainingSetDir}/trainingSetStrat1.csv")
   # End of Body of writing dataframe to file for trainingSet1. 
   return



"""
[Will be addressed later!]
def func2():
    # [Strat 2]The share price-to-NCAV criterion is sometimes used as a standalone tool for identifying undervalued stocks. **Garham considered a company's NCAV to be one of the most accurate representations of a company's true intrinsic value**
    # Psuedosteps: a) Creating a python list of independent variables, b) Using the list of indep variables to create Series, c) Copying the respective series to the column ids required to support model's decisions, d) Removing independent vars from the original dataframe , e) Load the resultant dataset to the testSet file[or create a new one? Go into more detail about this later], f) Use an algorithm that applies the order of optimality based on the strategy to columns on dataframe and then Write the resultant dataframe to testSet file. 
   indepVariables = []
   DepVariables = []
   resultantDataFrame = pd.DataFrame()
   companies=[]
   indepVariables: str = [] #<-- When filling in indepVars, make sure to refer to contents of the ticker object provided by yfinance. FOr more info, refer to this link and filepath respectively: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html#yfinance.Ticker and /c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/asyncNotes/lecture3.md.[UPDATE: After doing researhc, found that the following obj attrs coantin essential info for value investing: a) `balance_sheet`, b) `cash_flow`, c) `financials`, d) `history`. Prepending `quarterly_` to the aforemtneioned attrs allows us to get the quarterly basis, "allowing for more granular analysis". 
   # PN: May be wise to do this process iteratively via a for loop since each company will have thier respective values. OR, it can possibly be done using shortcuts provided by pandas and numpy. part 2) Heavily believe that this shoudl be implemnted via for loop to go through each company and add the rows iteratively. 
   DepVariables = []
   resultantDataFrame = pd.DataFrame()
   stratConditionsForOptimalValStock = []
   # Body of assigning new columns to dataframe 
   resultantDataFrame["Company"] = pd.Series(companies) #<-- Adds each company's tuple
   resultantDataFrame["P/B"] = mainDataSet[indepVariables[0]]/mainDataSet[indepVariables[1]]
   # (cont here)


   # End of assigning new columns to dataframe 

   # Body of adding column to dataframe to create trainingSet  



   # End of Body of writing resultant Dataframe to testSet
   # Body of writing resultant Dataframe to testSet
     filepathToTestSetDir: str
     mainDataset.to_csv(f"{filepathToTestSetDir}/testSetStrat2.csv")
   # End of Body of writing resultant Dataframe to testSet
   # Body of adding column to dataframe to create trainingSet  

   # End of body of adding column to dataframe to create trainingSet  
   # Body of writing dataframe to file for trainingSet1. 
     filepathToTrainingSetDir: str
     mainDataset.to_csv(f"{filepathToTrainingSetDir}/trainingSetStrat2.csv")
   # End of Body of writing dataframe to file for trainingSet1. 
    return

def func3():
    # [Strat 3] DCF Analysis uses future free cash flow (FCF) projections and discount weights that are calculated using the **Weighted Average Cost of Capitla (WACC)** to estimate the **present value of a company, with the underlying idea being that its intrinsic value is largely dependent on the company's ability to generate cash flow**. The essential calculation of a DCF analysis is as follows: **Fair Value = The Company's Enterprise Value - The Company's Debt** (**Enterprise value** is an alternative methic to market capitalization value. It represetns **market capitalization  + debt + preferred shared - total cash**, including cash equivalents). If the DCF analysis of a company renders a per-share value higher than the current share price, then the stock is considered undervalued.
    # Psuedosteps: a) Creating a python list of independent variables, b) Using the list of indep variables to create Series, c) Copying the respective series to the column ids required to support model's decisions, d) Removing independent vars from the original dataframe , e) Load the resultant dataset to the testSet file[or create a new one? Go into more detail about this later], f) Use an algorithm that applies the order of optimality based on the strategy to columns on dataframe and then Write the resultant dataframe to testSet file. 
   indepVariables = []
   DepVariables = []
   resultantDataFrame = pd.DataFrame()
   companies=[]
   indepVariables: str = [] #<-- When filling in indepVars, make sure to refer to contents of the ticker object provided by yfinance. FOr more info, refer to this link and filepath respectively: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html#yfinance.Ticker and /c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/asyncNotes/lecture3.md.[UPDATE: After doing researhc, found that the following obj attrs coantin essential info for value investing: a) `balance_sheet`, b) `cash_flow`, c) `financials`, d) `history`. Prepending `quarterly_` to the aforemtneioned attrs allows us to get the quarterly basis, "allowing for more granular analysis". 
   # PN: May be wise to do this process iteratively via a for loop since each company will have thier respective values. OR, it can possibly be done using shortcuts provided by pandas and numpy. part 2) Heavily believe that this shoudl be implemnted via for loop to go through each company and add the rows iteratively. 
   DepVariables = []
   resultantDataFrame = pd.DataFrame()
   stratConditionsForOptimalValStock = []
   # Body of assigning new columns to dataframe 
   resultantDataFrame["Company"] = pd.Series(companies) #<-- Adds each company's tuple
   resultantDataFrame["P/B"] = mainDataSet[indepVariables[0]]/mainDataSet[indepVariables[1]]
   # (cont here)


   # End of assigning new columns to dataframe 

   # Body of adding column to dataframe to create trainingSet  



   # End of Body of writing resultant Dataframe to testSet
   # Body of writing resultant Dataframe to testSet
     filepathToTestSetDir: str
     resultantDataFrame.to_csv(f"{filepathToTestSetDir}/testSetStrat4.csv")
   # End of Body of writing resultant Dataframe to testSet
   # Body of adding column to dataframe to create trainingSet  

   # End of body of adding column to dataframe to create trainingSet  
   # Body of writing dataframe to file for trainingSet1. 
     filepathToTrainingSetDir: str
     resultantDataFrame.to_csv(f"{filepathToTrainingSetDir}/trainingSetStrat4.csv")
   # End of Body of writing dataframe to file for trainingSet1. 
    return

def func4():

    return
def func5():
    # [Strat 6] The formula for calculating the Ben Graham Number is as follows: **Ben Graham Number = The Square Root of [22.5 x (Earnings per Share (EPS)) x (Book Value per Share)]**. For example, the Ben Graham Number for a stokc with an EPS of \$1.50 and a book value of \$10 per share calculates out to $\sqrt{22.5 \times 1.5 \times 10} = 18.37$. **Graham generally felt that a company's P/E ratio shouldn't be higher than 15 and that its P/B ratio shoulnd't exceed 1.5**. THat's where the 22.5 in the formula is derived from(15 x 1.5 = 22.5). However, with the valuation levels that are commonplace thes edays, the maximum allowable P/E might be shifted to around 25 . a) If the current share price is lower than the Ben Graham Number, this indicates the stock is undervalued and may be considered as a buy. b) If the current share price is higher than the Ben Graham Number, then the stock appears overvalued and not a promising buy candidate. 
    # Psuedosteps: a) Creating a python list of independent variables, b) Using the list of indep variables to create Series, c) Copying the respective series to the column ids required to support model's decisions, d) Removing independent vars from the original dataframe , e) Load the resultant dataset to the testSet file[or create a new one? Go into more detail about this later], f) Use an algorithm that applies the order of optimality based on the strategy to columns on dataframe and then Write the resultant dataframe to testSet file. 
   indepVariables = []
   DepVariables = []
   resultantDataFrame = pd.DataFrame()
   # Body of assigning new columns to dataframe 


   # End of assigning new columns to dataframe 

   # Body of adding column to dataframe to create trainingSet  



   # End of Body of writing resultant Dataframe to testSet
   # Body of adding column to dataframe to create trainingSet  

   # End of body of adding column to dataframe to create trainingSet  
    return


# End of Body of the functions used to exercise the substrats of value investing. Will dictate the contents of the trainingSet and testSet. 

# (cont here by writing other code that'll be needed)
func1()
"""
